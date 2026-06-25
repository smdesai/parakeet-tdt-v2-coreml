import AVFoundation
import Foundation

/// Decodes a (possibly security-scoped) file URL to mono Float32 PCM at the given
/// sample rate. Chunked AVAudioConverter loop with a final flush — the iOS
/// file-import counterpart of the CLI's `AudioLoader` (which takes a path).
/// Ported from parakeet-unified's AudioFileSamples.
enum AudioFileSamples {
    static func load(url: URL, sampleRate: Double) throws -> [Float] {
        let scoped = url.startAccessingSecurityScopedResource()
        defer {
            if scoped { url.stopAccessingSecurityScopedResource() }
        }

        let file = try AVAudioFile(forReading: url)
        let inputFormat = file.processingFormat
        guard
            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            ),
            let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
        else {
            throw NSError(
                domain: "AudioFileSamples",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported audio file format"]
            )
        }

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let inputCapacity = AVAudioFrameCount(min(max(file.length, 1), 32768))
        let outputCapacity = AVAudioFrameCount(Double(inputCapacity) * ratio + 4096)
        guard
            let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputCapacity),
            let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: max(outputCapacity, 4096))
        else {
            throw NSError(
                domain: "AudioFileSamples",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Could not allocate audio conversion buffer"]
            )
        }

        var samples: [Float] = []
        samples.reserveCapacity(Int(Double(file.length) * ratio) + 64)

        func appendOutput() {
            guard output.frameLength > 0, let channel = output.floatChannelData else { return }
            samples.append(
                contentsOf: UnsafeBufferPointer(start: channel[0], count: Int(output.frameLength))
            )
        }

        while file.framePosition < file.length {
            let remaining = AVAudioFrameCount(file.length - file.framePosition)
            try file.read(into: input, frameCount: min(inputCapacity, remaining))
            guard input.frameLength > 0 else { break }

            var consumed = false
            var needsMoreOutput = true
            while needsMoreOutput {
                output.frameLength = 0
                var error: NSError?
                let status = converter.convert(to: output, error: &error) { _, outStatus in
                    if consumed {
                        outStatus.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    outStatus.pointee = .haveData
                    return input
                }
                if status == .error {
                    throw error ?? NSError(
                        domain: "AudioFileSamples",
                        code: -4,
                        userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"]
                    )
                }
                appendOutput()
                needsMoreOutput = status == .haveData
            }
        }

        var flushing = true
        while flushing {
            output.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, outStatus in
                outStatus.pointee = .endOfStream
                return nil
            }
            if status == .error {
                throw error ?? NSError(
                    domain: "AudioFileSamples",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: "Audio conversion failed"]
                )
            }
            appendOutput()
            flushing = status == .haveData
        }

        guard !samples.isEmpty else {
            throw NSError(
                domain: "AudioFileSamples",
                code: -5,
                userInfo: [NSLocalizedDescriptionKey: "Audio file produced no samples"]
            )
        }

        return samples
    }
}
