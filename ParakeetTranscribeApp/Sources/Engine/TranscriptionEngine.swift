import CoreML
import Foundation
import SwiftUI

/// Drives the bundled Parakeet-TDT-v2 CoreML pipeline behind a SwiftUI-friendly
/// ObservableObject. Two entry points:
///   • `transcribeFile(_:)` — decode an imported audio file, then run the batch
///     Strategy-C `Transcriber` (one continuous decode over the whole clip).
///   • mic — `toggleRecording()` captures audio as a stream and feeds it to the
///     incremental `StreamingTranscriber`, which runs the SAME sliding window
///     progressively: it finalizes one window as soon as that window's full 15 s
///     span (center + 2.5 s right-context) is recorded, so the transcript grows
///     on screen and the result matches the batch path.
/// Model load and inference run on a serial background queue; published state is
/// always mutated on the main actor.
@MainActor
final class TranscriptionEngine: ObservableObject {

    enum Phase: Equatable {
        case idle
        case downloading(DownloadProgress)   // NEW — first-run model fetch
        case preparing                       // compile + map weights
        case ready
        case transcribing
        case recording
        case failed(String)
    }

    struct BatchMetrics: Equatable {
        var rtfx: Double
        var audioDuration: TimeInterval
        var transcriptionDuration: TimeInterval
        var backendLabel: String
        /// Per-stage CoreML busy time (preprocess/encode/decode/joint). Lets us see
        /// whether the 3-stage pipeline overlaps: wall ≈ max stage → overlapping;
        /// wall ≈ sum of stages → serialized. Empty for the mic path.
        var stages: [Profile.Stage] = []
    }

    @Published private(set) var phase: Phase = .idle
    @Published var transcript: String = ""
    @Published private(set) var batchMetrics: BatchMetrics?
    @Published private(set) var level: Float = 0

    /// Compute-unit label shown in the header (the models run on `.all`).
    let backendLabel = "CoreML · ANE"

    // Derived flags used by the views.
    var isPreparing: Bool   { phase == .preparing }
    var isTranscribing: Bool { phase == .transcribing }
    var isRecording: Bool   { phase == .recording }
    var isReady: Bool       { if case .ready = phase { return true }; return false }
    var isDownloading: Bool { if case .downloading = phase { return true }; return false }

    /// The latest download progress snapshot, or nil when not downloading.
    var downloadProgress: DownloadProgress? {
        if case .downloading(let p) = phase { return p }
        return nil
    }

    // Heavy objects live on the worker queue; never touched from the main actor
    // after construction.
    private var transcriber: Transcriber?
    private var runner: ModelRunner?
    private var tokenizer: ParakeetTokenizer?
    private let worker = DispatchQueue(label: "com.sdesai.parakeet.transcribe", qos: .userInitiated)

    // Fetches the model bundle from Hugging Face on first launch; no-op once
    // the on-disk `.complete` sentinel is present.
    private let downloader = ModelDownloader()

    // Mic streaming state.
    private let recorder = MicRecorder()
    private var recordingTask: Task<Void, Never>?

    // MARK: - model preparation

    /// Download the model bundle if needed, then load the models once. Safe to
    /// call repeatedly (no-op when ready and a no-restart when a download is
    /// already in flight). Retry from the failure overlay re-enters here and
    /// resumes the download (already-downloaded files are skipped by size).
    func prepareIfNeeded() async {
        switch phase {
        case .ready, .downloading, .preparing, .transcribing, .recording: return
        default: break
        }
        do {
            // First-run model fetch. `ensureInstalled` returns immediately once
            // the `.complete` sentinel is present, so normal launches skip this.
            if !ModelDownloader.isInstalled() {
                phase = .downloading(.zero)
                try await downloader.ensureInstalled { [weak self] p in
                    self?.phase = .downloading(p)
                }
            }
            // Compile + map the (now on-disk) models.
            phase = .preparing
            let loaded = try await loadModels()
            self.runner = loaded.runner
            self.tokenizer = loaded.tokenizer
            self.transcriber = loaded.transcriber
            phase = .ready
        } catch {
            phase = .failed(describe(error))
        }
    }

    private struct Loaded {
        let runner: ModelRunner
        let tokenizer: ParakeetTokenizer
        let transcriber: Transcriber
    }

    private func loadModels() async throws -> Loaded {
        try await withCheckedThrowingContinuation { cont in
            worker.async {
                do {
                    let modelsDir = try Self.modelsDir()
                    let vocabURL = modelsDir.appendingPathComponent("parakeet_vocab.json")
                    let runner = try ModelRunner(modelsDir: modelsDir, computeUnits: .all)
                    let tokenizer = try ParakeetTokenizer(contentsOf: vocabURL)
                    let t = Transcriber(
                        runner: runner,
                        tokenizer: tokenizer,
                        ctxSamples: Const.defaultCtx,
                        modelsDir: modelsDir,
                        primaryIsCPU: false
                    )
                    cont.resume(returning: Loaded(runner: runner, tokenizer: tokenizer, transcriber: t))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// The single production seam that resolves where models live. Models are
    /// downloaded from Hugging Face on first launch into a writable directory
    /// under Application Support (see `ModelDownloader`); that install root is
    /// the flat `modelsDir` the rest of the pipeline consumes.
    private static func modelsDir() throws -> URL {
        let dir = ModelDownloader.rootDirectory()
        let encoder = dir.appendingPathComponent("parakeet_encoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: encoder.path) else {
            throw EngineError.missingModels(
                "model download incomplete at \(dir.path) — re-launch the app to resume the download")
        }
        return dir
    }

    // MARK: - file transcription (batch Strategy C)

    func transcribeFile(_ url: URL) async {
        guard isReady else { return }
        phase = .transcribing
        transcript = ""
        batchMetrics = nil
        do {
            let samples = try await loadFile(url)
            guard !samples.isEmpty else {
                phase = .failed("No audio samples to transcribe.")
                return
            }
            guard let transcriber else {
                phase = .failed("Models are not loaded.")
                return
            }
            let audioDuration = Double(samples.count) / Double(Const.sampleRate)
            Profile.reset()
            let start = Date()
            let text = try await runBatch(samples, with: transcriber)
            let elapsed = Date().timeIntervalSince(start)
            let stages = Profile.snapshot()

            transcript = text
            batchMetrics = BatchMetrics(
                rtfx: elapsed > 0 ? audioDuration / elapsed : 0,
                audioDuration: audioDuration,
                transcriptionDuration: elapsed,
                backendLabel: "\(backendLabel) · file",
                stages: stages
            )
            phase = .ready
        } catch {
            phase = .failed(describe(error))
        }
    }

    private func loadFile(_ url: URL) async throws -> [Float] {
        try await withCheckedThrowingContinuation { cont in
            worker.async {
                do {
                    let s = try AudioFileSamples.load(url: url, sampleRate: Double(Const.sampleRate))
                    cont.resume(returning: s)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func runBatch(_ samples: [Float], with transcriber: Transcriber) async throws -> String {
        try await withCheckedThrowingContinuation { cont in
            worker.async {
                do { cont.resume(returning: try transcriber.transcribe(samples)) }
                catch { cont.resume(throwing: error) }
            }
        }
    }

    // MARK: - mic streaming (incremental Strategy C)

    func toggleRecording() {
        isRecording ? stopRecording() : startRecording()
    }

    private func startRecording() {
        guard isReady, recordingTask == nil,
              let runner, let tokenizer else { return }
        transcript = ""
        batchMetrics = nil
        recordingTask = Task { [weak self] in
            await self?.runRecording(runner: runner, tokenizer: tokenizer)
            self?.recordingTask = nil
        }
    }

    private func stopRecording() {
        // Tear the mic down off the main thread (AVAudioSession deactivation can
        // block); finishing the stream lets the recording loop run its final flush.
        recorder.onLevel = nil
        let rec = recorder
        Task.detached { rec.stop() }
    }

    private func runRecording(runner: ModelRunner, tokenizer: ParakeetTokenizer) async {
        guard await MicRecorder.requestPermission() else {
            phase = .failed("Microphone access denied. Enable it in Settings to record.")
            return
        }

        // Build the incremental transcriber on the worker queue (its init primes
        // the predictor with a CoreML call).
        let stream: AsyncStream<[Float]>
        let streamer: StreamingTranscriber
        do {
            streamer = try await makeStreamer(runner: runner, tokenizer: tokenizer)
            recorder.onLevel = { [weak self] lvl in
                Task { @MainActor in self?.level = lvl }
            }
            stream = try await Task.detached(priority: .userInitiated) { [recorder] in
                try recorder.start()
            }.value
        } catch {
            phase = .failed(describe(error))
            recorder.onLevel = nil
            return
        }

        phase = .recording
        let startAudio = Date()

        for await chunk in stream {
            do {
                let text = try await feed(chunk, to: streamer, isFinal: false)
                transcript = text
            } catch {
                phase = .failed(describe(error))
                break
            }
        }

        // Final flush: emit the trailing window (no right-context), matching the
        // batch path's last window.
        if case .recording = phase {
            do {
                let text = try await feed([], to: streamer, isFinal: true)
                transcript = text
                let audioDuration = await streamerSeconds(streamer)
                let elapsed = Date().timeIntervalSince(startAudio)
                batchMetrics = BatchMetrics(
                    rtfx: elapsed > 0 ? audioDuration / elapsed : 0,
                    audioDuration: audioDuration,
                    transcriptionDuration: elapsed,
                    backendLabel: "\(backendLabel) · mic"
                )
                phase = .ready
            } catch {
                phase = .failed(describe(error))
            }
        }
        level = 0
    }

    private func makeStreamer(runner: ModelRunner, tokenizer: ParakeetTokenizer) async throws -> StreamingTranscriber {
        try await withCheckedThrowingContinuation { cont in
            worker.async {
                do {
                    let s = try StreamingTranscriber(runner: runner, tokenizer: tokenizer,
                                                     ctxSamples: Const.defaultCtx)
                    cont.resume(returning: s)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Append a chunk and advance the decode on the worker queue; returns the
    /// transcript so far.
    private func feed(_ chunk: [Float], to streamer: StreamingTranscriber, isFinal: Bool) async throws -> String {
        try await withCheckedThrowingContinuation { cont in
            worker.async {
                do {
                    if !chunk.isEmpty { streamer.appendAudio(chunk) }
                    cont.resume(returning: try streamer.transcribeAvailable(isFinal: isFinal))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    private func streamerSeconds(_ streamer: StreamingTranscriber) async -> Double {
        await withCheckedContinuation { cont in
            worker.async { cont.resume(returning: streamer.audioSeconds) }
        }
    }

    // MARK: - errors

    enum EngineError: Error, CustomStringConvertible {
        case missingModels(String)
        var description: String {
            switch self {
            case .missingModels(let m): return m
            }
        }
    }

    private func describe(_ error: Error) -> String {
        if let e = error as? CustomStringConvertible { return e.description }
        return error.localizedDescription
    }
}
