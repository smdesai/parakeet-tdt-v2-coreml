import CoreML
import Foundation

/// A single continuous encoder feature stream, C-major: value(channel, t) =
/// data[channel * frames + t]. This is the concatenation of every window's
/// emitted center-frame slice (spec §6.4) and is the input to the TDT decode.
struct EncodedStream {
    let data: [Float]   // [1, channels, frames] flattened C-major
    let channels: Int
    let frames: Int
}

/// Runs preprocessor + encoder over each planned window and extracts the emitted
/// center frames, concatenating them into one stream (spec §6).
struct Encoder {
    let runner: ModelRunner
    let planner: WindowPlanner

    /// One window's emitted slice, stored FRAME-major as [len, channels]:
    /// value(t, channel) = data[t * channels + channel]. Frame-major so the
    /// pipelined decode stage can bulk-append it into its frame-major buffer
    /// without a per-element transpose on the (contended) decode thread.
    struct Slice { let data: [Float]; let len: Int }

    /// One window after the preprocessor stage: the padded mel ready for the
    /// encoder, plus the spec needed to map encoder frames back to the emit region.
    /// Carried between the preprocess and encode pipeline stages.
    struct PreprocessedWindow { let melPadded: MLMultiArray; let melLength: Int; let spec: WindowSpec }

    func encode(_ wav: [Float]) throws -> EncodedStream {
        let channels = Const.encoderChannels
        var slices: [Slice] = []
        slices.reserveCapacity(planner.plan(nSamples: wav.count).count)
        try encodeStreaming(wav) { slice in slices.append(slice) }
        return concat(slices, channels: channels)
    }

    /// Encode each planned window and hand its emitted center slice (C-major
    /// [channels, len]) to `onSlice`, in window order. This is the single source
    /// of truth for the per-window encode math; `encode` collects + concatenates,
    /// while the pipelined path decodes each slice as it arrives. Slices are
    /// produced identically either way.
    func encodeStreaming(_ wav: [Float], onSlice: (Slice) throws -> Void) throws {
        for spec in planner.plan(nSamples: wav.count) {
            let pre = try preprocessWindow(wav, spec)
            if let slice = try encodeWindow(pre) { try onSlice(slice) }
        }
    }

    /// Stage 1 (CPU+GPU): build the right-zero-padded window segment, run the
    /// preprocessor, and right-pad the mel to the fixed encoder input.
    func preprocessWindow(_ wav: [Float], _ spec: WindowSpec) throws -> PreprocessedWindow {
        // Window samples, right-zero-padded to the fixed encoder window.
        var seg = [Float](repeating: 0, count: planner.winSamples)
        let end = min(spec.winStart + planner.winSamples, wav.count)
        if end > spec.winStart {
            for i in spec.winStart..<end { seg[i - spec.winStart] = wav[i] }
        }

        // preprocessor -> mel[1,128,T_mel]; pass the TRUE unpadded length.
        let mel = try runner.runPreprocessor(audio: seg, trueLen: spec.trueLen)

        // Right-pad mel to the fixed [1,128,1501] encoder input.
        let melPadded = try padMel(mel.mel, toFrames: Const.melFrames)
        return PreprocessedWindow(melPadded: melPadded, melLength: mel.melLength, spec: spec)
    }

    /// Stage 2 (ANE): run the encoder over one preprocessed window and extract its
    /// emitted center slice. Returns nil when the emit region is empty.
    func encodeWindow(_ pre: PreprocessedWindow) throws -> Slice? {
        let channels = Const.encoderChannels
        let enc = try runner.runEncoder(melPadded: pre.melPadded, melLength: pre.melLength)

        // Map the emitted sample region [centerLo,centerHi) to encoder frames
        // using this window's own samples->frames ratio (spec §6.3).
        let spec = pre.spec
        let tValid = enc.validFrames
        let ratio = Double(tValid) / Double(max(1, spec.trueLen))
        var fs = Int((Double(spec.centerLo - spec.winStart) * ratio).rounded())
        var fe = Int((Double(spec.centerHi - spec.winStart) * ratio).rounded())
        fs = max(0, min(fs, tValid))
        fe = max(fs, min(fe, tValid))
        guard fe > fs else { return nil }

        return extractSlice(from: enc, frameLo: fs, frameHi: fe, channels: channels)
    }

    /// Extract encoder frames [frameLo,frameHi) of `enc` (C-major [channels,frames])
    /// as a FRAME-major [len,channels] slice: out[t*channels + c] = enc(c, frameLo+t).
    /// The transpose runs here on the encode thread (which is otherwise blocked
    /// waiting on the ANE), so the decode thread can bulk-copy the slice instead of
    /// transposing it. Same values, just reordered — the decode is unaffected.
    private func extractSlice(from enc: ModelRunner.EncoderOut,
                              frameLo: Int, frameHi: Int, channels: Int) -> Slice {
        let len = frameHi - frameLo
        var out = [Float](repeating: 0, count: len * channels)
        for c in 0..<channels {
            let srcBase = c * enc.frames + frameLo
            for t in 0..<len { out[t * channels + c] = enc.data[srcBase + t] }
        }
        return Slice(data: out, len: len)
    }

    /// Concatenate per-window FRAME-major [len_i,channels] slices into one
    /// C-major [channels, ΣLen] stream (single allocation). Transposes back to
    /// C-major because `EncodedStream`/`TdtDecoder` consume C-major; only the
    /// baseline (`encode`) path uses this — the hot pipelined path keeps the
    /// frame-major slices as-is.
    private func concat(_ slices: [Slice], channels: Int) -> EncodedStream {
        let total = slices.reduce(0) { $0 + $1.len }
        guard total > 0 else { return EncodedStream(data: [], channels: channels, frames: 0) }

        var data = [Float](repeating: 0, count: channels * total)
        var frameOffset = 0
        for slice in slices {
            for t in 0..<slice.len {
                let srcBase = t * channels
                let dstT = frameOffset + t
                for c in 0..<channels { data[c * total + dstT] = slice.data[srcBase + c] }
            }
            frameOffset += slice.len
        }
        return EncodedStream(data: data, channels: channels, frames: total)
    }

    /// Right-pad (or trim) a mel array [1,128,T] to [1,128,toFrames] with zeros.
    private func padMel(_ mel: MLMultiArray, toFrames: Int) throws -> MLMultiArray {
        let shape = mel.shape.map { $0.intValue }
        let bins = shape.count >= 2 ? shape[shape.count - 2] : Const.melBins
        let t = shape.last ?? 0
        if t == toFrames { return mel }

        let flat = MLArray.floats(mel)            // C-major [1,bins,t]: data[b*t + i]
        var padded = [Float](repeating: 0, count: bins * toFrames)
        let copyT = min(t, toFrames)
        for b in 0..<bins {
            let srcBase = b * t
            let dstBase = b * toFrames
            for i in 0..<copyT { padded[dstBase + i] = flat[srcBase + i] }
        }
        let dtype = runner.encoder.modelDescription
            .inputDescriptionsByName["mel"]?.multiArrayConstraint?.dataType ?? .float32
        return MLArray.float(padded, shape: [1, bins, toFrames], dataType: dtype)
    }
}
