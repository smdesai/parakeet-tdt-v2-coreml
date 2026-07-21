import CoreML
import Foundation

/// Incremental (progressive) Strategy-C transcriber for the live-mic path.
///
/// The batch `Transcriber` plans every window up front and decodes once. This
/// class does the same work *as audio arrives*: it finalizes one window as soon
/// as that window's full 15 s span has been recorded, appends the window's emitted
/// center frames to a growing stream, and advances ONE continuous TDT decode over
/// the new frames (carrying the LSTM + emission state forward).
///
/// Why the output matches the batch path exactly (for audio > one window):
///   • A non-final window at center [c0, c0+center) is encoded only once its whole
///     `[winStart, winStart+winSamples)` span is recorded — i.e. with its real
///     2.5 s right-context, never zero-padded. That makes its encoder output
///     identical to the offline run's.
///   • The decode is the same single continuous pass; feeding frames in chunks
///     vs all at once is deterministically equivalent (a TDT duration skip that
///     lands past the currently-available frames simply pauses the loop until
///     more arrive — no frame is consumed early or twice).
/// Audio ≤ one window is deferred to a single full window on `finish()`, matching
/// `WindowPlanner`'s short-audio fast path.
///
/// NOT thread-safe: drive it from one serial queue (the engine's worker).
public final class StreamingTranscriber {
    let runner: ModelRunner
    let tokenizer: ParakeetTokenizer

    private let winSamples: Int
    private let ctxSamples: Int
    private let centerSamples: Int
    private let channels: Int

    // Accumulated mono 16 kHz audio (absolute sample indices are stable).
    private var wav: [Float] = []
    // Next center start to emit (absolute samples). 0 until the first window.
    private var nextCenter = 0

    // Continuous decode state ----------------------------------------------
    // Emitted center frames, FRAME-major: frame t at [t*channels, (t+1)*channels).
    private var frames: [Float] = []
    private var timeIdx = 0
    private var emitted: [Int] = []
    private var state: ModelRunner.DecoderState
    private var lastEmissionFrame = -1
    private var emissionsAtThisFrame = 0
    private let encStep: MLMultiArray
    private let melInputType: MLMultiArrayDataType

    public init(runner: ModelRunner, tokenizer: ParakeetTokenizer, ctxSamples: Int) throws {
        self.runner = runner
        self.tokenizer = tokenizer
        self.winSamples = Const.winSamples
        self.ctxSamples = ctxSamples
        self.centerSamples = Const.winSamples - 2 * ctxSamples
        precondition(centerSamples > 0, "ctxSamples=\(ctxSamples) too large for window \(Const.winSamples)")
        self.channels = Const.encoderChannels
        self.encStep = runner.makeEncoderStep()
        self.melInputType = runner.encoder.modelDescription
            .inputDescriptionsByName["mel"]?.multiArrayConstraint?.dataType ?? .float32

        // Prime the predictor with blank as start-of-sequence (once).
        self.state = runner.freshDecoderState()
        try runner.runDecoder(token: Const.blankId, state: &self.state)
    }

    /// Append newly-recorded samples (a delta, not the whole buffer).
    public func appendAudio(_ samples: [Float]) {
        wav.append(contentsOf: samples)
    }

    /// Total seconds of audio accumulated so far.
    public var audioSeconds: Double { Double(wav.count) / Double(Const.sampleRate) }

    /// The full accumulated buffer (for a final batch fallback if needed).
    var fullBuffer: [Float] { wav }

    /// Encode any windows whose full span is now recorded, decode the new frames,
    /// and return the transcript so far. Call with `isFinal: true` once recording
    /// has stopped to flush the trailing (right-context-less) window.
    public func transcribeAvailable(isFinal: Bool) throws -> String {
        let n = wav.count

        if isFinal {
            if nextCenter == 0 && n <= winSamples {
                // Short audio: a single full window, exactly like the batch path.
                try encodeWindow(winStart: 0, trueLen: n, centerLo: 0, centerHi: n)
                nextCenter = max(1, n)
            } else {
                // Emit every remaining center up to n; the last has no right context.
                while nextCenter < n {
                    let centerHi = min(nextCenter + centerSamples, n)
                    let winStart = max(0, nextCenter - ctxSamples)
                    let trueLen = min(winSamples, n - winStart)
                    try encodeWindow(winStart: winStart, trueLen: trueLen,
                                     centerLo: nextCenter, centerHi: centerHi)
                    nextCenter += centerSamples
                }
            }
        } else {
            // Streaming: only start once we've exceeded a single window (so we know
            // the offline plan is multi-window), then emit each FULL-center window
            // whose entire 15 s span is recorded (real right-context, no padding).
            guard n > winSamples else { return currentText() }
            while true {
                let centerHi = nextCenter + centerSamples
                let winStart = max(0, nextCenter - ctxSamples)
                guard centerHi <= n, winStart + winSamples <= n else { break }
                try encodeWindow(winStart: winStart, trueLen: winSamples,
                                 centerLo: nextCenter, centerHi: centerHi)
                nextCenter += centerSamples
            }
        }

        try advanceDecode()
        return currentText()
    }

    public func currentText() -> String { tokenizer.text(for: emitted) }

    // MARK: - per-window encode (mirrors Encoder.swift, frame-major output)

    private func encodeWindow(winStart: Int, trueLen: Int, centerLo: Int, centerHi: Int) throws {
        guard centerHi > centerLo else { return }

        // Window samples, right-zero-padded to the fixed encoder window.
        var seg = [Float](repeating: 0, count: winSamples)
        let end = min(winStart + winSamples, wav.count)
        if end > winStart {
            for i in winStart..<end { seg[i - winStart] = wav[i] }
        }

        let mel = try runner.runPreprocessor(audio: seg, trueLen: trueLen)
        let melPadded = try padMel(mel.mel, toFrames: Const.melFrames)
        let enc = try runner.runEncoder(melPadded: melPadded, melLength: mel.melLength)

        // Map the emitted sample region to encoder frames (this window's ratio).
        let tValid = enc.validFrames
        let ratio = Double(tValid) / Double(max(1, trueLen))
        var fs = Int((Double(centerLo - winStart) * ratio).rounded())
        var fe = Int((Double(centerHi - winStart) * ratio).rounded())
        fs = max(0, min(fs, tValid))
        fe = max(fs, min(fe, tValid))
        guard fe > fs else { return }

        // Transpose C-major enc.data [channels, frames] -> frame-major append.
        frames.reserveCapacity(frames.count + (fe - fs) * channels)
        for t in fs..<fe {
            for c in 0..<channels {
                frames.append(enc.data[c * enc.frames + t])
            }
        }
    }

    /// Right-pad (or trim) a mel array [1,128,T] to [1,128,toFrames] (cf. Encoder.padMel).
    private func padMel(_ mel: MLMultiArray, toFrames: Int) throws -> MLMultiArray {
        let shape = mel.shape.map { $0.intValue }
        let bins = shape.count >= 2 ? shape[shape.count - 2] : Const.melBins
        let t = shape.last ?? 0
        if t == toFrames { return mel }

        let flat = MLArray.floats(mel)
        var padded = [Float](repeating: 0, count: bins * toFrames)
        let copyT = min(t, toFrames)
        for b in 0..<bins {
            let srcBase = b * t
            let dstBase = b * toFrames
            for i in 0..<copyT { padded[dstBase + i] = flat[srcBase + i] }
        }
        return MLArray.float(padded, shape: [1, bins, toFrames], dataType: melInputType)
    }

    // MARK: - continuous TDT decode (mirrors TdtDecoder.decode, resumable)

    private func advanceDecode() throws {
        let avail = frames.count / channels
        let durBins = Const.durationBins
        @inline(__always) func mapDuration(_ bin: Int) -> Int {
            (bin >= 0 && bin < durBins.count) ? durBins[bin] : bin
        }

        while timeIdx < avail {
            fillStep(frame: timeIdx)
            let decision = try runner.runJoint(encoderStep: encStep, decoderOutput: state.output)
            let tok = decision.tokenId
            var dur = mapDuration(decision.duration)

            if tok == Const.blankId {
                if dur == 0 { dur = 1 }            // force progress
                timeIdx += dur
            } else {
                if dur == 0 && timeIdx == lastEmissionFrame && emissionsAtThisFrame >= 1 {
                    dur = 1
                }
                let emitFrame = timeIdx
                emitted.append(tok)
                try runner.runDecoder(token: tok, state: &state)
                timeIdx += dur

                if emitFrame == lastEmissionFrame {
                    emissionsAtThisFrame += 1
                } else {
                    lastEmissionFrame = emitFrame
                    emissionsAtThisFrame = 1
                }
                if emissionsAtThisFrame >= Const.maxSymbolsPerStep {
                    timeIdx += 1
                    emissionsAtThisFrame = 0
                    lastEmissionFrame = -1
                }
            }
        }
    }

    /// Fill the reusable [1,channels,1] encoder-step from the FRAME-major buffer:
    /// frame t's channel vector is contiguous at [t*channels, (t+1)*channels).
    private func fillStep(frame: Int) {
        let base = frame * channels
        switch encStep.dataType {
        case .float16:
            encStep.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float16.self)
                for ch in 0..<channels { dst[ch] = Float16(frames[base + ch]) }
            }
        case .float32:
            encStep.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float32.self)
                for ch in 0..<channels { dst[ch] = frames[base + ch] }
            }
        default:
            for ch in 0..<channels { encStep[ch] = NSNumber(value: frames[base + ch]) }
        }
    }
}
