import CoreML
import Foundation

/// Single continuous TDT decode over the concatenated center-frame stream.
/// Semantics ported from FluidAudio's TdtDecoderV3 (spec §7), simplified to one
/// non-streaming, non-batched, eager-emission pass.
///
/// TDT (Token-and-Duration Transducer) differs from plain RNN-T: each joint step
/// predicts a token AND a duration (frames to skip). Duration advances the time
/// pointer on EVERY step — blank and non-blank alike — which is the key
/// difference from the plain-RNN-T loop in parakeet-unified's CoreMLParakeet.swift.
struct TdtDecoder {
    let runner: ModelRunner

    /// Optional keyword booster (shallow fusion). When nil or empty, the decode is
    /// byte-identical to the baseline (plain `runJoint` argmax path).
    var booster: KeywordBooster? = nil

    /// Decode the whole stream into emitted text token ids (blanks filtered out).
    func decode(_ stream: EncodedStream) throws -> [Int] {
        guard stream.frames > 0 else { return [] }

        var timeIdx = 0
        var tokens: [Int] = []

        // Keyword-boost bookkeeping. `generated` is the parallel Int32 token history
        // the booster matches against; `boosting` is resolved once so the hot loop
        // never pays for the optional/empty check per frame.
        var generated: [Int32] = []
        let boosting = (booster.map { !$0.isEmpty }) ?? false

        // Per-FRAME emission tracking (NOT a global counter). The force-blank guard
        // must only fire when too many tokens land on the *same* frame; counting
        // emissions globally would wrongly skip frames during dense speech and drop
        // tokens. See TdtDecoderV3 lines 446–462.
        var lastEmissionFrame = -1
        var emissionsAtThisFrame = 0

        // Prime the predictor with blank as start-of-sequence.
        var state = runner.freshDecoderState()
        try runner.runDecoder(token: Const.blankId, state: &state)

        let encStep = runner.makeEncoderStep()
        let durBins = Const.durationBins

        @inline(__always) func mapDuration(_ bin: Int) -> Int {
            (bin >= 0 && bin < durBins.count) ? durBins[bin] : bin
        }

        while timeIdx < stream.frames {
            runner.fillEncoderStep(encStep, from: stream.data,
                                   channels: stream.channels, frames: stream.frames, frame: timeIdx)
            let tok: Int
            var dur: Int
            if boosting {
                // Boosted path: read raw logits, add keyword bonuses, then argmax.
                let step = try runner.runJointLogits(encoderStep: encStep, decoderOutput: state.output)
                let bonuses = booster!.nextTokenBonuses(generated: generated)
                tok = MLArray.argmax(logits: step.logits, count: Const.vocabSize + 1, bonuses: bonuses)
                dur = mapDuration(step.duration)
            } else {
                let decision = try runner.runJoint(encoderStep: encStep, decoderOutput: state.output)
                tok = decision.tokenId
                dur = mapDuration(decision.duration)
            }

            if tok == Const.blankId {
                // Blank: advance time, do NOT update the decoder LSTM (silence
                // carries no language context). Force progress so we can't stall.
                if dur == 0 { dur = 1 }
                timeIdx += dur
            } else {
                // Prevent repeated non-blank emissions stuck on the same frame when
                // duration == 0 (canonical lines 308–314).
                if dur == 0 && timeIdx == lastEmissionFrame && emissionsAtThisFrame >= 1 {
                    dur = 1
                }

                let emitFrame = timeIdx
                tokens.append(tok)
                if boosting { generated.append(Int32(tok)) }       // extend booster history
                try runner.runDecoder(token: tok, state: &state)   // update predictor
                timeIdx += dur                                     // TDT: token also skips `dur` frames

                // Update per-frame emission count.
                if emitFrame == lastEmissionFrame {
                    emissionsAtThisFrame += 1
                } else {
                    lastEmissionFrame = emitFrame
                    emissionsAtThisFrame = 1
                }

                // Force-blank guard: too many tokens at one frame -> force advance.
                if emissionsAtThisFrame >= Const.maxSymbolsPerStep {
                    timeIdx += 1
                    emissionsAtThisFrame = 0
                    lastEmissionFrame = -1
                }
            }
        }
        return tokens
    }
}

/// Resumable form of the same TDT decode, for the pipelined (encode∥decode) path.
/// Frames arrive a window-slice at a time (C-major [channels, len]) and are
/// appended to a FRAME-major buffer (frame t at [t*channels, (t+1)*channels)).
/// `decodeAvailable()` advances the *identical* state machine as `TdtDecoder`
/// over whatever frames have arrived, pausing when a duration skip lands past the
/// frontier and resuming when more frames are appended. The token sequence is
/// therefore identical to the one-shot decode — only the scheduling differs.
///
/// NOT thread-safe: append + decode must run on a single consumer thread.
final class StreamingTdtDecoder {
    private let runner: ModelRunner
    private let channels: Int

    /// Optional keyword booster (shallow fusion). nil/empty => byte-identical to the
    /// baseline (plain `runJoint` argmax path). See `KeywordBooster`.
    var booster: KeywordBooster? = nil

    private var frames: [Float] = []   // FRAME-major
    private var availFrames = 0
    private var timeIdx = 0
    private var tokens: [Int] = []
    private var generated: [Int32] = []   // parallel booster history (boosted path)
    private var state: ModelRunner.DecoderState
    private var lastEmissionFrame = -1
    private var emissionsAtThisFrame = 0
    private let encStep: MLMultiArray
    private let durBins = Const.durationBins

    init(runner: ModelRunner, booster: KeywordBooster? = nil) throws {
        self.runner = runner
        self.booster = booster
        self.channels = Const.encoderChannels
        self.encStep = runner.makeEncoderStep()
        self.state = runner.freshDecoderState()
        // Prime the predictor with blank as start-of-sequence (matches TdtDecoder).
        try runner.runDecoder(token: Const.blankId, state: &self.state)
    }

    /// Append one window's emitted slice into the frame-major buffer. The slice is
    /// already FRAME-major ([len,channels], transposed on the encode thread), so the
    /// per-frame channel vectors are contiguous and this is a flat bulk copy — no
    /// per-element transpose on this (contended) decode thread. `fillStep` then reads
    /// each frame as a flat run.
    func append(_ slice: Encoder.Slice) {
        let len = slice.len
        guard len > 0 else { return }
        frames.append(contentsOf: slice.data)
        availFrames += len
    }

    @inline(__always) private func mapDuration(_ bin: Int) -> Int {
        (bin >= 0 && bin < durBins.count) ? durBins[bin] : bin
    }

    /// Advance the decode over all currently-available frames. Safe to call
    /// repeatedly as more slices are appended.
    func decodeAvailable() throws {
        let boosting = (booster.map { !$0.isEmpty }) ?? false
        while timeIdx < availFrames {
            fillStep(frame: timeIdx)
            let tok: Int
            var dur: Int
            if boosting {
                let step = try runner.runJointLogits(encoderStep: encStep, decoderOutput: state.output)
                let bonuses = booster!.nextTokenBonuses(generated: generated)
                tok = MLArray.argmax(logits: step.logits, count: Const.vocabSize + 1, bonuses: bonuses)
                dur = mapDuration(step.duration)
            } else {
                let decision = try runner.runJoint(encoderStep: encStep, decoderOutput: state.output)
                tok = decision.tokenId
                dur = mapDuration(decision.duration)
            }

            if tok == Const.blankId {
                if dur == 0 { dur = 1 }
                timeIdx += dur
            } else {
                if dur == 0 && timeIdx == lastEmissionFrame && emissionsAtThisFrame >= 1 {
                    dur = 1
                }
                let emitFrame = timeIdx
                tokens.append(tok)
                if boosting { generated.append(Int32(tok)) }
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

    /// Flush the final decode (all frames have arrived) and return the tokens.
    func finish() throws -> [Int] {
        try decodeAvailable()
        return tokens
    }

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
