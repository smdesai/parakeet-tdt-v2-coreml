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

    /// Advance the predictor with one token, mutating `state` (output/h/c).
    func runDecoder(token: Int, state: inout DecoderState) throws {
        let out = try predict(decoder, [
            "targets": MLArray.int32([Int32(token)], shape: [1, 1]),
            "target_length": MLArray.int32([1], shape: [1]),
            "h_in": state.h,
            "c_in": state.c,
        ])
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

    /// `encoderStep` is a reusable [1,1024,1] array filled by the caller.
    func runJoint(encoderStep: MLMultiArray, decoderOutput: [Float]) throws -> JointDecision {
        let decStep = MLArray.float(decoderOutput, shape: [1, Const.decoderHidden, 1], dataType: decStepType)
        let out = try predict(joint, ["encoder_step": encoderStep, "decoder_step": decStep])
        guard
            let tok = out.featureValue(for: "token_id")?.multiArrayValue,
            let dur = out.featureValue(for: "duration")?.multiArrayValue
        else { throw ModelError.missingOutput("token_id/duration") }
        return JointDecision(tokenId: MLArray.firstInt(tok), duration: MLArray.firstInt(dur))
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
        if Profile.enabled {
            let label = modelLabel(model)
            let t0 = Date()
            let r = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: dict))
            Profile.record(label, Date().timeIntervalSince(t0))
            return r
        }
        return try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: dict))
    }

    private func modelLabel(_ model: MLModel) -> String {
        if model === preprocessor { return "preprocessor" }
        if model === encoder { return "encoder" }
        if model === decoder { return "decoder" }
        if model === joint { return "joint" }
        return "other"
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
