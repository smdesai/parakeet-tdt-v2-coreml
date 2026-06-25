import Foundation

/// Verified constants for Parakeet-TDT-v2 0.6B CoreML. Do not change without
/// re-deriving from the `.nemo` / `.mlmodelc` metadata — see docs/swift-port-spec.md.
enum Const {
    static let sampleRate      = 16_000
    static let winSamples      = 240_000          // 15 s — fixed encoder window
    static let defaultCtx      = 40_000           // 2.5 s look-around each side
    static let melBins         = 128
    static let melFrames       = 1_501            // fixed encoder mel width
    static let encoderChannels = 1_024
    static let encoderMaxFrames = 188             // static upper bound; trust encoder_length
    static let decoderHidden   = 640
    static let decoderLayers   = 2                // h/c are [2,1,640]

    // TDT decode
    static let vocabSize         = 1_024          // text tokens 0…1023
    static let blankId           = 1_024          // == vocabSize; never rendered as text
    static let durationBins      = [0, 1, 2, 3, 4] // num_extra_outputs = 5
    static let maxSymbolsPerStep = 10             // emissions cap per time index
}
