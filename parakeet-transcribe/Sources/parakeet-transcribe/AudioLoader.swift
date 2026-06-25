import AVFoundation
import Foundation

enum AudioError: Error, CustomStringConvertible {
    case open(String)
    case convert(String)
    var description: String {
        switch self {
        case .open(let m):    return "audio open failed: \(m)"
        case .convert(let m): return "audio convert failed: \(m)"
        }
    }
}

/// Decodes any AVFoundation-readable file to mono Float32 PCM at 16 kHz.
/// Multi-channel input is downmixed (channel average); any sample rate is
/// resampled. Output samples are in [-1, 1] — the preprocessor performs NeMo's
/// mel normalization internally, so no extra scaling is applied here (spec §4).
enum AudioLoader {
    static func load(path: String, sampleRate: Int = Const.sampleRate) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioError.open(error.localizedDescription)
        }

        let srcFormat = file.processingFormat
        guard let dstFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw AudioError.convert("could not build 16 kHz mono format")
        }

        // Fast path: already mono @ target rate, no conversion needed.
        if srcFormat.sampleRate == Double(sampleRate) && srcFormat.channelCount == 1 {
            return try readDirect(file: file)
        }

        guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
            throw AudioError.convert("no converter \(srcFormat) -> \(dstFormat)")
        }

        let srcFrames = AVAudioFrameCount(file.length)
        guard srcFrames > 0 else { return [] }
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrames) else {
            throw AudioError.convert("could not allocate input buffer")
        }
        do {
            try file.read(into: inBuf)
        } catch {
            throw AudioError.convert("read: \(error.localizedDescription)")
        }

        // Output capacity scaled by the resample ratio (+ slack).
        let ratio = dstFormat.sampleRate / srcFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 4096
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: outCapacity) else {
            throw AudioError.convert("could not allocate output buffer")
        }

        var fed = false
        var convErr: NSError?
        let status = converter.convert(to: outBuf, error: &convErr) { _, outStatus in
            if fed {
                outStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            outStatus.pointee = .haveData
            return inBuf
        }
        if let convErr { throw AudioError.convert(convErr.localizedDescription) }
        if status == .error { throw AudioError.convert("converter returned .error") }

        return floatChannel(outBuf)
    }

    /// Read a file that is already mono @ target rate.
    private static func readDirect(file: AVAudioFile) throws -> [Float] {
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0 else { return [] }
        guard let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames) else {
            throw AudioError.convert("could not allocate buffer")
        }
        try file.read(into: buf)
        return floatChannel(buf)
    }

    /// Extract channel 0 of a non-interleaved Float32 buffer as `[Float]`.
    private static func floatChannel(_ buf: AVAudioPCMBuffer) -> [Float] {
        let n = Int(buf.frameLength)
        guard n > 0, let ch = buf.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
    }
}
