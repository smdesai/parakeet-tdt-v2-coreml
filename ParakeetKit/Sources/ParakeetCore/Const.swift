import Foundation

/// Verified constants for Parakeet-TDT-v2 0.6B CoreML. Do not change without
/// re-deriving from the `.nemo` / `.mlmodelc` metadata — see docs/swift-port-spec.md.
public enum Const {
    public static let sampleRate      = 16_000
    public static let winSamples      = 240_000          // 15 s — fixed encoder window
    public static let defaultCtx      = 40_000           // 2.5 s look-around each side
    public static let melBins         = 128
    public static let melFrames       = 1_501            // fixed encoder mel width
    public static let encoderChannels = 1_024
    public static let encoderMaxFrames = 188             // static upper bound; trust encoder_length
    public static let decoderHidden   = 640
    public static let decoderLayers   = 2                // h/c are [2,1,640]

    // TDT decode
    public static let vocabSize         = 1_024          // text tokens 0…1023
    public static let blankId           = 1_024          // == vocabSize; never rendered as text
    public static let durationBins      = [0, 1, 2, 3, 4] // num_extra_outputs = 5
    public static let maxSymbolsPerStep = 10             // emissions cap per time index
}
