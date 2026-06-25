// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "parakeet-transcribe",
    platforms: [
        .macOS(.v14)        // CoreML iOS17+/macOS14+ target; fixed-shape models
    ],
    targets: [
        .executableTarget(
            name: "parakeet-transcribe",
            path: "Sources/parakeet-transcribe"
        ),
        .testTarget(
            name: "parakeet-transcribeTests",
            dependencies: ["parakeet-transcribe"],
            path: "Tests/parakeet-transcribeTests"
        ),
    ]
)
