import CoreML
import Foundation

/// Ties encoder + TDT decode + detokenize into the two strategies (spec §3).
final class Transcriber {
    let runner: ModelRunner
    let tokenizer: ParakeetTokenizer
    let ctxSamples: Int

    /// Where to load the CPU fallback runner from, and whether the primary path
    /// is already CPU (in which case the fallback is pointless and skipped).
    let modelsDir: URL
    let primaryIsCPU: Bool

    /// Optional keyword booster (shallow fusion). nil => baseline decode, wired
    /// unchanged into every strategy and the CPU fallback.
    let booster: KeywordBooster?

    /// Lazily-built CPU(fp32) runner, used only on the rare empty-output retry.
    private var cpuRunner: ModelRunner?

    init(runner: ModelRunner, tokenizer: ParakeetTokenizer, ctxSamples: Int,
         modelsDir: URL, primaryIsCPU: Bool, booster: KeywordBooster? = nil) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.ctxSamples = ctxSamples
        self.modelsDir = modelsDir
        self.primaryIsCPU = primaryIsCPU
        self.booster = booster
    }

    /// Strategy C (recommended): overlapping windows + ONE continuous TDT decode.
    func transcribe(_ wav: [Float]) throws -> String {
        let ids = try decodeStrategyC(wav, with: runner)

        // Rare ANE/GPU fp16 collapse: when speech begins softly after leading
        // silence, the first-token-vs-blank decision at the onset frame is a
        // near-tie. fp16 rounding (ANE *and* GPU both flip; only CPU fp32 holds)
        // tips it to blank; because the predictor is frozen at SOS until the
        // first emission, that single flip cascades into an all-blank decode →
        // empty output (~1 in 1400 real clips; see docs/ane-onset-collapse.md).
        // The empty result is an unambiguous signal: retry once on a CPU runner.
        if ids.isEmpty && !primaryIsCPU {
            let fallback = try cpuFallbackRunner()
            let ids2 = try decodeStrategyC(wav, with: fallback)
            return tokenizer.text(for: ids2)
        }
        return tokenizer.text(for: ids)
    }

    private func decodeStrategyC(_ wav: [Float], with runner: ModelRunner) throws -> [Int] {
        let planner = WindowPlanner(ctxSamples: ctxSamples)
        // Three-stage pipeline, one engine per stage, so all three stay busy:
        //   stage 1 (preprocess, CPU+GPU) -> stage 2 (encode, ANE) -> stage 3
        //   (TDT decode, CPU) on this thread.
        // Previously stage 1+2 ran serially on one producer thread, leaving the ANE
        // idle for every ~15ms preprocess; splitting them lets the encoder run
        // back-to-back so the per-window preprocess hides under it. The windows are
        // produced in the same order with the same values, and the resumable
        // decoder runs the identical state machine, so the emitted token sequence
        // is byte-identical to `Encoder.encode` + `TdtDecoder.decode`.
        let encoder = Encoder(runner: runner, planner: planner)
        let melQueue = BoundedQueue<Encoder.PreprocessedWindow>(capacity: 4)
        let sliceQueue = BoundedQueue<Encoder.Slice>(capacity: 8)
        let decoder = try StreamingTdtDecoder(runner: runner, booster: booster)

        // Stage 1: preprocess every planned window, in order.
        let preThread = Thread {
            do {
                for spec in planner.plan(nSamples: wav.count) {
                    melQueue.push(try encoder.preprocessWindow(wav, spec))
                }
                melQueue.finish(error: nil)
            } catch {
                melQueue.finish(error: error)
            }
        }
        preThread.stackSize = 8 << 20
        preThread.start()

        // Stage 2: encode each preprocessed window (ANE), feeding the decode stage.
        let encThread = Thread {
            do {
                while let pre = melQueue.pop() {
                    if let slice = try encoder.encodeWindow(pre) { sliceQueue.push(slice) }
                }
                sliceQueue.finish(error: melQueue.producerError)
            } catch {
                sliceQueue.finish(error: error)
            }
        }
        encThread.stackSize = 8 << 20
        encThread.start()

        // Stage 3: decode each slice as it arrives on this thread.
        while let slice = sliceQueue.pop() {
            decoder.append(slice)
            try decoder.decodeAvailable()
        }
        if let err = sliceQueue.producerError { throw err }
        return try decoder.finish()
    }

    /// Build (once) and return a CPU-only runner for the empty-output retry.
    private func cpuFallbackRunner() throws -> ModelRunner {
        if let r = cpuRunner { return r }
        let r = try ModelRunner(modelsDir: modelsDir, computeUnits: .cpuOnly)
        cpuRunner = r
        return r
    }

    /// Strategy A (baseline): non-overlapping 15s windows, decode each from blank,
    /// string-join. Kept for comparison (spec §11). Implemented as ctx=0 windows,
    /// each decoded independently as its own stream.
    func transcribeBaseline(_ wav: [Float]) throws -> String {
        let planner = WindowPlanner(ctxSamples: 0)         // center == full window
        let encoder = Encoder(runner: runner, planner: planner)
        var decoder = TdtDecoder(runner: runner)
        decoder.booster = booster
        var parts: [String] = []
        for spec in planner.plan(nSamples: wav.count) {
            // Encode just this one window as a standalone stream.
            let single = sliceWav(wav, spec: spec, winSamples: planner.winSamples)
            let stream = try encoder.encode(single)
            let ids = try decoder.decode(stream)
            let txt = tokenizer.text(for: ids).trimmingCharacters(in: .whitespaces)
            if !txt.isEmpty { parts.append(txt) }
        }
        return parts.joined(separator: " ")
    }

    private func sliceWav(_ wav: [Float], spec: WindowSpec, winSamples: Int) -> [Float] {
        let end = min(spec.winStart + winSamples, wav.count)
        guard end > spec.winStart else { return [] }
        return Array(wav[spec.winStart..<end])
    }
}

/// Bounded blocking hand-off of items from a single producer thread to a single
/// consumer thread. `push` blocks when full so the producer can't run unboundedly
/// ahead of the consumer (caps memory); `pop` blocks until an item is available or
/// the producer finishes. Used to chain the preprocess→encode→decode stages.
final class BoundedQueue<T> {
    private let capacity: Int
    private var items: [T] = []
    private var head = 0          // amortized-O(1) dequeue without removeFirst shift
    private var done = false
    private(set) var producerError: Error?
    private let cond = NSCondition()

    init(capacity: Int) { self.capacity = max(1, capacity) }

    func push(_ item: T) {
        cond.lock()
        while (items.count - head) >= capacity && !done { cond.wait() }
        if !done { items.append(item) }
        cond.signal()
        cond.unlock()
    }

    func finish(error: Error?) {
        cond.lock()
        producerError = error
        done = true
        cond.broadcast()
        cond.unlock()
    }

    /// Returns the next item, or nil once the producer has finished and drained.
    func pop() -> T? {
        cond.lock()
        while head >= items.count && !done { cond.wait() }
        let item: T? = head < items.count ? items[head] : nil
        if item != nil {
            head += 1
            if head > 64 && head * 2 >= items.count {   // compact occasionally
                items.removeFirst(head); head = 0
            }
        }
        cond.signal()
        cond.unlock()
        return item
    }
}
