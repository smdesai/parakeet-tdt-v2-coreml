import CoreML
import Foundation

enum ModelError: Error, CustomStringConvertible {
    case load(String)
    case missingOutput(String)
    var description: String {
        switch self {
        case .load(let m):          return "model load failed: \(m)"
        case .missingOutput(let m): return "model produced no output \(m)"
        }
    }
}

/// Loads the four runtime `.mlmodelc` models and exposes typed wrappers for each.
/// All float I/O is Float32 and all int I/O is Int32 per the verified metadata
/// (spec §1.1, §9), but the dtype is queried per input so the code survives a
/// re-export to fp16. Outputs are read stride-aware (ANE buffers can be padded).
final class ModelRunner {
    let preprocessor: MLModel
    let encoder: MLModel
    let decoder: MLModel
    let joint: MLModel

    init(modelsDir: URL, computeUnits: MLComputeUnits = .all) throws {
        // Per-model compute-unit override for diagnosis, e.g.
        //   PARAKEET_CU_parakeet_encoder=cpu  forces just the encoder onto CPU.
        func cuOverride(_ name: String) -> MLComputeUnits? {
            guard let v = ProcessInfo.processInfo.environment["PARAKEET_CU_\(name)"] else { return nil }
            switch v.lowercased() {
            case "cpu": return .cpuOnly
            case "cpugpu", "cpu_gpu": return .cpuAndGPU
            case "ane", "cpuane", "cpu_ane": return .cpuAndNeuralEngine
            case "all": return .all
            default: return nil
            }
        }
        func load(_ name: String) throws -> MLModel {
            let url = modelsDir.appendingPathComponent(name + ".mlmodelc")
            let cfg = MLModelConfiguration()
            cfg.computeUnits = cuOverride(name) ?? computeUnits
            do {
                return try MLModel(contentsOf: url, configuration: cfg)
            } catch {
                throw ModelError.load("\(name).mlmodelc: \(error.localizedDescription)")
            }
        }
        self.preprocessor = try load("parakeet_preprocessor")
        self.encoder      = try load("parakeet_encoder")
        self.decoder      = try load("parakeet_decoder")
        self.joint        = try load("parakeet_joint_decision_single_step")
    }

    // MARK: typed dtype lookups (queried once)

    private func inType(_ model: MLModel, _ name: String) -> MLMultiArrayDataType {
        model.modelDescription.inputDescriptionsByName[name]?
            .multiArrayConstraint?.dataType ?? .float32
    }

    // MARK: preprocessor  audio_signal[1,N] f32 + audio_length[1] i32 -> mel[1,128,T] + mel_length[1]

    struct MelOut { let mel: MLMultiArray; let melLength: Int }

    func runPreprocessor(audio: [Float], trueLen: Int) throws -> MelOut {
        let n = audio.count
        let aType = inType(preprocessor, "audio_signal")
        let audioArr = MLArray.float(audio, shape: [1, n], dataType: aType)
        let lenArr = MLArray.int32([Int32(trueLen)], shape: [1])
        let out = try predict(preprocessor, ["audio_signal": audioArr, "audio_length": lenArr])
        guard let mel = out.featureValue(for: "mel")?.multiArrayValue else {
            throw ModelError.missingOutput("mel")
        }
        let melLen = out.featureValue(for: "mel_length")?.multiArrayValue.map { MLArray.firstInt($0) } ?? mel.shape.last!.intValue
        return MelOut(mel: mel, melLength: melLen)
    }

    // MARK: encoder  mel[1,128,1501] f32 + mel_length[1] i32 -> encoder[1,1024,188] + encoder_length[1]

    struct EncoderOut {
        let data: [Float]   // logical C-contiguous [1,1024,T]: value(c,t)=data[c*frames+t]
        let channels: Int
        let frames: Int     // static frame dim of the buffer
        let validFrames: Int // encoder_length (≤ frames)
    }

    func runEncoder(melPadded: MLMultiArray, melLength: Int) throws -> EncoderOut {
        let lenArr = MLArray.int32([Int32(melLength)], shape: [1])
        let out = try predict(encoder, ["mel": melPadded, "mel_length": lenArr])
        guard let enc = out.featureValue(for: "encoder")?.multiArrayValue else {
            throw ModelError.missingOutput("encoder")
        }
        let shape = enc.shape.map { $0.intValue }
        let channels = shape.count >= 2 ? shape[shape.count - 2] : Const.encoderChannels
        let frames = shape.last ?? Const.encoderMaxFrames
        let encLen = out.featureValue(for: "encoder_length")?.multiArrayValue.map { MLArray.firstInt($0) } ?? frames
        return EncoderOut(data: MLArray.floats(enc), channels: channels, frames: frames,
                          validFrames: min(encLen, frames))
    }

    // MARK: decoder  targets[1,1] + target_length[1] + h_in[2,1,640] + c_in[2,1,640]
    //                -> decoder[1,640,1] + h_out + c_out

    struct DecoderState {
        var h: MLMultiArray
        var c: MLMultiArray
        var output: [Float]   // length = decoderHidden, logical [1,640,1]
    }

    func freshDecoderState() -> DecoderState {
        let hType = inType(decoder, "h_in")
        let cType = inType(decoder, "c_in")
        let zeros = [Float](repeating: 0, count: Const.decoderLayers * Const.decoderHidden)
        return DecoderState(
            h: MLArray.float(zeros, shape: [Const.decoderLayers, 1, Const.decoderHidden], dataType: hType),
            c: MLArray.float(zeros, shape: [Const.decoderLayers, 1, Const.decoderHidden], dataType: cType),
            output: [Float](repeating: 0, count: Const.decoderHidden))
    }

    // Reusable provider + input buffers for the decoder hot path (FeatureBag). The
    // decode thread is the *only* caller (spec §pipeline stage 3), so a single
    // per-runner bag is safe — see runJoint's note for the threading argument.
    // `targets` is overwritten in place each call; `target_length` is a constant 1
    // written once at construction; `h_in`/`c_in` are fresh model outputs each step
    // (their object identity changes), so the bag swaps in those two MLFeatureValues
    // per call rather than copying into a fixed buffer (same data either way).
    private lazy var decoderTargets: MLMultiArray =
        MLArray.int32([0], shape: [1, 1])               // overwritten in place each call
    private lazy var decoderBag: FeatureBag = {
        let bag = FeatureBag(values: [
            "targets": MLFeatureValue(multiArray: decoderTargets),
            "target_length": MLFeatureValue(multiArray: MLArray.int32([1], shape: [1])),
        ])
        return bag
    }()

    /// Advance the predictor with one token, mutating `state` (output/h/c).
    func runDecoder(token: Int, state: inout DecoderState) throws {
        // Overwrite the reused targets buffer in place (int32 [1,1]).
        decoderTargets.withUnsafeMutableBytes { ptr, _ in
            ptr.bindMemory(to: Int32.self)[0] = Int32(token)
        }
        // h_in/c_in are last step's model outputs (new objects each call) — set them
        // on the bag each call. target_length is constant and already set.
        decoderBag.set("h_in", MLFeatureValue(multiArray: state.h))
        decoderBag.set("c_in", MLFeatureValue(multiArray: state.c))
        let out = try predict(decoder, provider: decoderBag)
        guard
            let dec = out.featureValue(for: "decoder")?.multiArrayValue,
            let h = out.featureValue(for: "h_out")?.multiArrayValue,
            let c = out.featureValue(for: "c_out")?.multiArrayValue
        else { throw ModelError.missingOutput("decoder/h_out/c_out") }
        state.output = MLArray.floats(dec)
        state.h = h
        state.c = c
    }

    // MARK: joint_decision_single_step
    //   encoder_step[1,1024,1] + decoder_step[1,640,1] -> token_id[1,1,1] + token_prob + duration[1,1,1]
    // The model argmaxes internally; we read token_id / duration directly (spec §1.1).

    struct JointDecision { let tokenId: Int; let duration: Int }

    private lazy var encStepType = inType(joint, "encoder_step")
    private lazy var decStepType = inType(joint, "decoder_step")

    // Reusable provider + input buffers for the joint hot path (FeatureBag). This is
    // the per-encoder-frame call (~50k/run) and the biggest win: zero per-call dict /
    // provider / array allocation. Single-threaded: runJoint is only ever called from
    // the decode thread (TdtDecoder / StreamingTdtDecoder, spec §pipeline stage 3), so
    // one per-runner reusable bag is safe. `decoderStep` is overwritten in place each
    // call with the SAME dtype-aware logic as the old `MLArray.float(...)`; `encoderStep`
    // is rebound into the bag whenever the caller's buffer object changes (see runJoint).
    private lazy var jointDecStep: MLMultiArray =
        MLArray.empty(shape: [1, Const.decoderHidden, 1], dataType: decStepType)
    private lazy var jointBag: FeatureBag = {
        FeatureBag(values: ["decoder_step": MLFeatureValue(multiArray: jointDecStep)])
    }()
    private var jointBoundEncStep: MLMultiArray?

    /// `encoderStep` is a reusable [1,1024,1] array filled by the caller.
    func runJoint(encoderStep: MLMultiArray, decoderOutput: [Float]) throws -> JointDecision {
        // Bind the caller's encoder_step buffer into the bag whenever the object
        // identity changes. Strategy C passes the SAME reused encStep on every call
        // (so this rebinds only once and the hot path stays allocation-free); the
        // `--baseline` path (TdtDecoder.decode) mints a fresh encStep per window, so
        // each window must rebind — otherwise the joint would keep reading the
        // previous window's stale features (a non-byte-identical regression).
        if encoderStep !== jointBoundEncStep {
            jointBag.set("encoder_step", MLFeatureValue(multiArray: encoderStep))
            jointBoundEncStep = encoderStep
        }
        // Overwrite the reused decoder_step buffer in place. This mirrors
        // MLArray.float(decoderOutput, ...) exactly (same Float16/Float32 conversion,
        // same element order) so the values fed to the model are byte-identical.
        fillDecStep(jointDecStep, from: decoderOutput)
        let out = try predict(joint, provider: jointBag)
        guard
            let tok = out.featureValue(for: "token_id")?.multiArrayValue,
            let dur = out.featureValue(for: "duration")?.multiArrayValue
        else { throw ModelError.missingOutput("token_id/duration") }
        return JointDecision(tokenId: MLArray.firstInt(tok), duration: MLArray.firstInt(dur))
    }

    /// Overwrite a [1,640,1] decoder_step buffer in place from `decoderOutput`,
    /// matching MLArray.float's dtype-aware conversion exactly (byte-identical).
    private func fillDecStep(_ arr: MLMultiArray, from values: [Float]) {
        let count = values.count
        switch arr.dataType {
        case .float16:
            arr.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float16.self)
                for i in 0..<count { dst[i] = Float16(values[i]) }
            }
        case .float32:
            arr.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float32.self)
                for i in 0..<count { dst[i] = values[i] }
            }
        default:
            for i in 0..<count { arr[i] = NSNumber(value: values[i]) }
        }
    }

    func makeEncoderStep() -> MLMultiArray {
        MLArray.empty(shape: [1, Const.encoderChannels, 1], dataType: encStepType)
    }

    /// Fill a reusable [1,1024,1] encoder-step array from a frame of the flat
    /// C-major encoder buffer: value(channel) = data[channel * frames + frame].
    func fillEncoderStep(_ step: MLMultiArray, from data: [Float], channels: Int, frames: Int, frame: Int) {
        MLArray.fillChannelStep(step, from: data, channels: channels, frames: frames, frame: frame)
    }

    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        let dict = inputs.mapValues { MLFeatureValue(multiArray: $0) }
        return try predict(model, provider: try MLDictionaryFeatureProvider(dictionary: dict))
    }

    /// Predict from a pre-built provider (the FeatureBag hot path), preserving the
    /// Profile timing wrapper and per-model labels.
    private func predict(_ model: MLModel, provider: MLFeatureProvider) throws -> MLFeatureProvider {
        if Profile.enabled {
            let label = modelLabel(model)
            let t0 = Date()
            let r = try model.prediction(from: provider)
            Profile.record(label, Date().timeIntervalSince(t0))
            return r
        }
        return try model.prediction(from: provider)
    }

    private func modelLabel(_ model: MLModel) -> String {
        if model === preprocessor { return "preprocessor" }
        if model === encoder { return "encoder" }
        if model === decoder { return "decoder" }
        if model === joint { return "joint" }
        return "other"
    }
}

/// A reusable `MLFeatureProvider` ("FeatureBag") that holds a fixed set of named
/// `MLFeatureValue`s, each wrapping a pre-allocated `MLMultiArray`. Overwriting an
/// array's contents in place (or swapping an entry via `set`) feeds new data to the
/// next `prediction(from:)` with zero per-call dictionary/provider allocation — the
/// hot-loop optimization for the joint (~per-frame) and decoder (~per-token) paths.
///
/// NOT thread-safe: each bag is owned by a single thread (the decode thread). The
/// joint and decoder bags are only ever used from that one thread (spec §pipeline).
final class FeatureBag: NSObject, MLFeatureProvider {
    private var values: [String: MLFeatureValue]

    init(values: [String: MLFeatureValue]) {
        self.values = values
    }

    /// Replace the stored value for a key (used for h_in/c_in, which are fresh model
    /// outputs each step and therefore new objects).
    func set(_ name: String, _ value: MLFeatureValue) {
        values[name] = value
    }

    var featureNames: Set<String> { Set(values.keys) }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

/// Opt-in per-model timing, enabled with PARAKEET_PROFILE=1. Accumulates wall time
/// and call counts per model so we can see where the run actually spends time.
/// Zero overhead when disabled (the `enabled` flag is read once).
enum Profile {
    static let enabled = ProcessInfo.processInfo.environment["PARAKEET_PROFILE"] == "1"
    private static var times: [String: Double] = [:]
    private static var counts: [String: Int] = [:]
    private static let lock = NSLock()

    static func record(_ label: String, _ seconds: Double) {
        lock.lock(); defer { lock.unlock() }
        times[label, default: 0] += seconds
        counts[label, default: 0] += 1
    }

    static func report() {
        guard enabled else { return }
        lock.lock(); defer { lock.unlock() }
        let total = times.values.reduce(0, +)
        var lines = ["--- model profile (PARAKEET_PROFILE) ---"]
        for label in times.keys.sorted(by: { times[$0]! > times[$1]! }) {
            let t = times[label]!, c = counts[label]!
            lines.append(String(format: "  %-12@ %8.3fs  %6d calls  %5.2fms/call  %5.1f%%",
                                label as NSString, t, c, t / Double(max(1, c)) * 1000,
                                total > 0 ? t / total * 100 : 0))
        }
        lines.append(String(format: "  %-12@ %8.3fs in-model total", "TOTAL" as NSString, total))
        FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }
}
