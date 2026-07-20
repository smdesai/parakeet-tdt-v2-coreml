import CoreML
import Foundation

// parakeet-transcribe — standalone long-form Parakeet-TDT-v2 transcriber.
//   parakeet-transcribe --models <dir> --audio <file> [--ctx-s 2.5] [--baseline]
// Implements Strategy C (overlapping windows + one continuous TDT decode).
// See docs/swift-port-spec.md.

struct Args {
    var modelsDir: String?
    var audio: String?
    var ctxSeconds: Double = 2.5
    var baseline = false
    var computeUnits: MLComputeUnits = .all
    var noTranscript = false
    var keywordsFile: String?
    var alpha: Double = 2.0
}

func parseArgs(_ argv: [String]) -> Args {
    var a = Args()
    var i = 0
    while i < argv.count {
        let arg = argv[i]
        func next() -> String? { i + 1 < argv.count ? argv[i + 1] : nil }
        switch arg {
        case "--models":   a.modelsDir = next(); i += 1
        case "--audio":    a.audio = next(); i += 1
        case "--ctx-s":    if let v = next(), let d = Double(v) { a.ctxSeconds = d }; i += 1
        case "--baseline": a.baseline = true
        case "--no-transcript": a.noTranscript = true
        case "--keywords": a.keywordsFile = next(); i += 1
        case "--alpha":    if let v = next(), let d = Double(v) { a.alpha = d }; i += 1
        case "--compute":
            if let v = next() {
                switch v.lowercased() {
                case "cpu": a.computeUnits = .cpuOnly
                case "cpugpu", "cpu_gpu": a.computeUnits = .cpuAndGPU
                case "ane", "cpuane", "cpu_ane": a.computeUnits = .cpuAndNeuralEngine
                default: a.computeUnits = .all
                }
            }
            i += 1
        case "-h", "--help":
            printUsage(); exit(0)
        default:
            break
        }
        i += 1
    }
    return a
}

func printUsage() {
    let usage = """
    parakeet-transcribe — long-form Parakeet-TDT-v2 CoreML transcriber

    USAGE:
      parakeet-transcribe --models <dir> --audio <file> [options]

    OPTIONS:
      --models <dir>    Directory with the 4 .mlmodelc models + parakeet_vocab.json
      --audio <file>    Audio file to transcribe (any AVFoundation format)
      --ctx-s <sec>     Context seconds each side of a window (default 2.5)
      --baseline        Use Strategy A (per-window decode + join) instead of C
      --compute <u>     Compute units: all | cpu | cpugpu | ane (default all)
      --no-transcript   Don't print the transcript; only emit timing/RTFx
                        (to stderr). Use when benchmarking RTFx.
      --keywords <file> One keyword/phrase per line (blank lines and #comments
                        skipped) to bias decoding toward (shallow fusion). Needs
                        the re-exported parakeet_joint_logits_single_step.mlmodelc
                        (see docs/word-boost.md); no-op without --keywords.
      --alpha <f>       Keyword boost strength (default 2.0). Larger = stronger bias.
      -h, --help        Show this help

    Timing (RTFx = audio seconds / transcription seconds) is always printed to
    stderr after a run, so stdout stays clean for the transcript text.
    """
    print(usage)
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data(("error: " + msg + "\n").utf8))
    exit(1)
}

let args = parseArgs(Array(CommandLine.arguments.dropFirst()))

guard let modelsDirPath = args.modelsDir else { fail("--models is required (see --help)") }
guard let audioPath = args.audio else { fail("--audio is required (see --help)") }

let modelsDir = URL(fileURLWithPath: modelsDirPath, isDirectory: true)
let vocabURL = modelsDir.appendingPathComponent("parakeet_vocab.json")
guard FileManager.default.fileExists(atPath: vocabURL.path) else {
    fail("parakeet_vocab.json not found in \(modelsDirPath) — extract it from the .nemo (see spec §1.2)")
}

do {
    let runner = try ModelRunner(modelsDir: modelsDir, computeUnits: args.computeUnits)
    let tokenizer = try ParakeetTokenizer(contentsOf: vocabURL)
    let wav = try AudioLoader.load(path: audioPath)
    if wav.isEmpty { fail("decoded 0 samples from \(audioPath)") }

    // Optional keyword booster: one keyword/phrase per line; blank lines and
    // #comments skipped. Each keyword is greedily encoded to token ids; keywords
    // that encode to nothing are dropped. If nothing survives, the booster is nil
    // and the decode is exactly the baseline.
    var booster: KeywordBooster? = nil
    if let kwPath = args.keywordsFile {
        let contents = try String(contentsOfFile: kwPath, encoding: .utf8)
        let keywords = contents.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        let seqs = keywords.map { tokenizer.encode($0) }.filter { !$0.isEmpty }
        booster = seqs.isEmpty ? nil : KeywordBooster(keywordTokenSeqs: seqs, alpha: Float(args.alpha))
        let msg = "keywords: \(keywords.count) read, \(seqs.count) armed (alpha \(args.alpha))\n"
        FileHandle.standardError.write(Data(msg.utf8))
    }

    let ctxSamples = max(0, Int((args.ctxSeconds * Double(Const.sampleRate)).rounded()))
    let transcriber = Transcriber(runner: runner, tokenizer: tokenizer, ctxSamples: ctxSamples,
                                  modelsDir: modelsDir, primaryIsCPU: args.computeUnits == .cpuOnly,
                                  booster: booster)

    // Time only the transcription itself (model + audio load excluded) so RTFx
    // reflects the decode pipeline, not one-time setup.
    let audioSeconds = Double(wav.count) / Double(Const.sampleRate)
    let start = Date()
    let text = args.baseline ? try transcriber.transcribeBaseline(wav)
                             : try transcriber.transcribe(wav)
    let elapsed = Date().timeIntervalSince(start)
    let rtfx = elapsed > 0 ? audioSeconds / elapsed : 0

    if !args.noTranscript {
        print(text)
    }

    // Metrics go to stderr so stdout stays clean for the transcript text (the WER
    // pipelines capture stdout).
    let metrics = String(
        format: "RTFx %.2fx · audio %.2fs · transcribe %.2fs",
        rtfx, audioSeconds, elapsed)
    FileHandle.standardError.write(Data((metrics + "\n").utf8))
    Profile.report()
} catch {
    fail("\(error)")
}
