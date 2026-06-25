import AVFoundation
import Foundation

/// Captures microphone audio as a stream of mono Float32 chunks at the model
/// sample rate (16 kHz). Each tap buffer is resampled and yielded immediately so
/// the engine can feed it to the incremental sliding-window decoder as it arrives
/// (rather than waiting for the recording to stop). Also reports a live RMS level
/// for the waveform meter.
final class MicRecorder {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let targetFormat: AVAudioFormat
    private var continuation: AsyncStream<[Float]>.Continuation?

    /// Callback with the latest input level in [0, 1] (RMS), for the waveform.
    var onLevel: ((Float) -> Void)?

    init(sampleRate: Double = Double(Const.sampleRate)) {
        targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!
    }

    /// Request mic permission (async wrapper over the AVAudioApplication callback).
    static func requestPermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return true
        case .denied:  return false
        case .undetermined:
            return await withCheckedContinuation { cont in
                AVAudioApplication.requestRecordPermission { granted in
                    cont.resume(returning: granted)
                }
            }
        @unknown default: return false
        }
    }

    /// Begin capture and return a stream of 16 kHz mono sample chunks. The stream
    /// finishes when `stop()` is called. Buffering is unbounded so no audio is
    /// dropped while a heavy encoder window is in flight.
    func start() throws -> AsyncStream<[Float]> {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        let (stream, continuation) = AsyncStream<[Float]>.makeStream(bufferingPolicy: .unbounded)
        self.continuation = continuation

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self, let chunk = self.resample(buffer), !chunk.isEmpty else { return }
            self.continuation?.yield(chunk)
        }

        engine.prepare()
        try engine.start()
        return stream
    }

    /// Stop capture and finish the stream.
    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        continuation?.finish()
        continuation = nil
        onLevel?(0)
    }

    // MARK: - tap resampling

    private func resample(_ buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let converter else { return nil }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return nil }

        var consumed = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }
        guard status != .error, out.frameLength > 0, let ch = out.floatChannelData else { return nil }

        let n = Int(out.frameLength)
        let ptr = UnsafeBufferPointer(start: ch[0], count: n)

        // RMS for the level meter.
        var sumSq: Float = 0
        for v in ptr { sumSq += v * v }
        let rms = (sumSq / Float(max(1, n))).squareRoot()
        onLevel?(min(1, rms * 4))

        return Array(ptr)
    }
}
