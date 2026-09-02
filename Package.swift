// swift-tools-version: 5.9
import PackageDescription

// ParakeetKit — the transcription core (and its Hugging Face model downloader)
// factored out of the parakeet-transcribe CLI and the ParakeetTranscribe iOS app
// so BOTH consume one library instead of each carrying a copy of the pipeline.
//
//   • ParakeetCore     — CoreML + Foundation only; the verified Parakeet-TDT-v2
//                        pipeline (preprocess → encode → TDT decode → detokenize),
//                        both the batch `Transcriber` and the incremental
//                        `StreamingTranscriber`. iOS + macOS.
//   • ParakeetDownload — the range-chunked parallel HF downloader used by the app
//                        on first launch. Foundation only (UIKit background-task
//                        assertion is behind `canImport(UIKit)`), so it also builds
//                        on macOS. Kept separate so the CLI can depend on the core
//                        without pulling in the networking/UIKit layer.
//
// Deployment floors match the two consumers: iOS 17 / macOS 14 (the CoreML
// fixed-shape models require iOS17+/macOS14+; the app itself targets a higher iOS
// but nothing here needs it).
//
// This manifest lives at the REPOSITORY ROOT so the package can be consumed as a
// git dependency (SwiftPM requires a root Package.swift); the Swift sources and
// tests stay under ParakeetKit/. In-repo consumers reference it as
// `.package(name: "ParakeetKit", path: "..")`.
let package = Package(
    name: "ParakeetKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "ParakeetCore", targets: ["ParakeetCore"]),
        .library(name: "ParakeetDownload", targets: ["ParakeetDownload"]),
    ],
    targets: [
        .target(name: "ParakeetCore", path: "ParakeetKit/Sources/ParakeetCore"),
        .target(name: "ParakeetDownload", path: "ParakeetKit/Sources/ParakeetDownload"),
        .testTarget(
            name: "ParakeetCoreTests", dependencies: ["ParakeetCore"],
            path: "ParakeetKit/Tests/ParakeetCoreTests"),
        .testTarget(
            name: "ParakeetDownloadTests", dependencies: ["ParakeetDownload"],
            path: "ParakeetKit/Tests/ParakeetDownloadTests"),
    ]
)
