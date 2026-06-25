import Foundation

/// One encoder window. The window feeds samples `[winStart, winStart+trueLen)`
/// (right-zero-padded to `winSamples`); only the center `[centerLo, centerHi)` is
/// emitted into the final decode stream.
struct WindowSpec: Equatable {
    let winStart: Int   // first sample fed to the preprocessor
    let trueLen:  Int   // real (unpadded) samples in the window  → audio_length
    let centerLo: Int   // first emitted sample (absolute)
    let centerHi: Int   // one-past-last emitted sample (absolute)
}

/// Sliding-window geometry for the fixed-15s encoder. Direct port of
/// `plan_windows` in `longform_transcribe.py` (spec §5).
///
/// The emitted **center** regions tile `[0, n)` contiguously and non-overlapping;
/// each window also reads `ctxSamples` of look-around on each side that is encoded
/// but NOT emitted. The first window has no left context (start of audio) and the
/// last has no right context (end of audio) — exactly what NeMo sees at the true
/// boundaries. Left padding is never used; the window start clamps to 0 and the
/// emit region shifts instead.
struct WindowPlanner {
    let winSamples: Int
    let ctxSamples: Int

    init(winSamples: Int = Const.winSamples, ctxSamples: Int = Const.defaultCtx) {
        precondition(winSamples - 2 * ctxSamples > 0,
                     "ctxSamples=\(ctxSamples) too large for window \(winSamples)")
        self.winSamples = winSamples
        self.ctxSamples = ctxSamples
    }

    /// Emitted center width in samples (= winSamples - 2*ctxSamples).
    var centerSamples: Int { winSamples - 2 * ctxSamples }

    func plan(nSamples n: Int) -> [WindowSpec] {
        // Short audio: single full-context window, no seam (matches the fast path).
        if n <= winSamples {
            return [WindowSpec(winStart: 0, trueLen: max(0, n), centerLo: 0, centerHi: max(0, n))]
        }

        var specs: [WindowSpec] = []
        let center = centerSamples
        var c0 = 0
        while c0 < n {
            let c1 = min(c0 + center, n)
            let winStart = max(0, c0 - ctxSamples)        // clamp, never left-pad
            let trueLen  = min(winSamples, n - winStart)
            specs.append(WindowSpec(winStart: winStart, trueLen: trueLen,
                                    centerLo: c0, centerHi: c1))
            c0 += center
        }
        return specs
    }
}
