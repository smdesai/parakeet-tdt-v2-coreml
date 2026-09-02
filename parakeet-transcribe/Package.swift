// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "parakeet-transcribe",
    platforms: [
        .macOS(.v14)        // CoreML iOS17+/macOS14+ target; fixed-shape models
    ],
    dependencies: [
        // The transcription core lives in the sibling ParakeetKit package, shared
        // with the iOS app. Consumed by local path (single source of truth).
        .package(name: "ParakeetKit", path: ".."),  // repo root hosts the ParakeetKit manifest
    ],
    targets: [
        .executableTarget(
            name: "parakeet-transcribe",
            dependencies: [
                .product(name: "ParakeetCore", package: "ParakeetKit"),
            ],
            path: "Sources/parakeet-transcribe"
        ),
    ]
)
