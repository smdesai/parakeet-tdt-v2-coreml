//
//  ModelDownloaderTests.swift
//  ParakeetTranscribeTests
//
//  Pure-logic tests for ModelDownloader's network-free planning seams.
//
//  These mirror parakeet-transcribe's WindowPlannerTests: plain XCTest over
//  deterministic logic, no CoreML, no live network, no filesystem. We exercise
//  ONLY externally observable behavior of the pure, `internal static` planning
//  functions:
//    - repoId(from:)             URL / bare-string -> "<org>/<name>"
//    - decodeTree(from:)         JSON -> [TreeEntry] (incl. nested lfs.size)
//    - planFiles(from:)          [TreeEntry] -> filtered + sorted [RemoteFile]
//    - isAllowed(path:)          allowlist predicate
//    - downloadURL(relativePath:) resolve-URL construction
//    - chunkGeometry(_:)         file size -> byte-range chunk plan
//    - canResume(...)            bitmap-resume decision
//
//  We never assert on the private transfer internals (fetchChunk / the session
//  pool / dataWithDeadline) — those are the impure network edges.
//

import XCTest
@testable import ParakeetTranscribe

final class ModelDownloaderTests: XCTestCase {

    // MARK: - Fixture (captured HF tree-API shape)
    //
    // A valid Hugging Face tree-API array for `smdesai/parakeet-tdt-0.6b-v2-coreml`.
    // Contents:
    //   - all directory entries (type "directory", size null),
    //   - the 21 runtime files with their EXACT tree-API sizes,
    //   - `.gitattributes` (1519)                        -> must be excluded (hidden),
    //   - a SYNTHETIC stray runtime artifact
    //     `parakeet_joint_logits_single_step.mlmodelc/weights/weight.bin` (6906368)
    //     -> must be excluded by the allowlist (proves future strays don't ship).
    //
    // Entries are deliberately scrambled (not pre-sorted) so the ascending-sort
    // assertion in testTreeToPlan is meaningful.
    private static let treeFixtureJSON = """
    [
      {"type": "file", "path": "parakeet_decoder.mlmodelc/metadata.json", "size": 3427},
      {"type": "directory", "path": "parakeet_preprocessor.mlmodelc", "size": null},
      {"type": "file", "path": "parakeet_decoder.mlmodelc/weights/weight.bin", "size": 14429952},
      {"type": "file", "path": "parakeet_encoder.mlmodelc/model.mil", "size": 1002653},
      {"type": "directory", "path": "parakeet_joint_decision_single_step.mlmodelc", "size": null},
      {"type": "file", "path": "parakeet_decoder.mlmodelc/model.mil", "size": 13106},
      {"type": "directory", "path": "parakeet_joint_logits_single_step.mlmodelc", "size": null},
      {"type": "directory", "path": "parakeet_preprocessor.mlmodelc/analytics", "size": null},
      {"type": "directory", "path": "parakeet_joint_logits_single_step.mlmodelc/weights", "size": null},
      {"type": "file", "path": "parakeet_preprocessor.mlmodelc/coremldata.bin", "size": 493},
      {"type": "file", "path": "parakeet_joint_decision_single_step.mlmodelc/model.mil", "size": 9722},
      {"type": "directory", "path": "parakeet_joint_decision_single_step.mlmodelc/weights", "size": null},
      {"type": "file", "path": "parakeet_preprocessor.mlmodelc/analytics/coremldata.bin", "size": 243},
      {"type": "directory", "path": "parakeet_decoder.mlmodelc/analytics", "size": null},
      {"type": "file", "path": "parakeet_encoder.mlmodelc/weights/weight.bin", "size": 594211328},
      {"type": "file", "path": "parakeet_preprocessor.mlmodelc/metadata.json", "size": 2953},
      {"type": "file", "path": "parakeet_preprocessor.mlmodelc/model.mil", "size": 26313},
      {"type": "directory", "path": "parakeet_encoder.mlmodelc/analytics", "size": null},
      {"type": "file", "path": "parakeet_decoder.mlmodelc/analytics/coremldata.bin", "size": 243},
      {"type": "file", "path": "parakeet_encoder.mlmodelc/metadata.json", "size": 2910},
      {"type": "file", "path": "parakeet_decoder.mlmodelc/coremldata.bin", "size": 554},
      {"type": "file", "path": "parakeet_encoder.mlmodelc/analytics/coremldata.bin", "size": 243},
      {"type": "file", "path": "parakeet_encoder.mlmodelc/coremldata.bin", "size": 478},
      {"type": "file", "path": "parakeet_joint_logits_single_step.mlmodelc/weights/weight.bin", "size": 6906368},
      {"type": "file", "path": "parakeet_joint_decision_single_step.mlmodelc/coremldata.bin", "size": 534},
      {"type": "file", "path": "parakeet_joint_decision_single_step.mlmodelc/metadata.json", "size": 2936},
      {"type": "file", "path": "parakeet_vocab.json", "size": 15519},
      {"type": "file", "path": ".gitattributes", "size": 1519},
      {"type": "directory", "path": "parakeet_preprocessor.mlmodelc/weights", "size": null},
      {"type": "file", "path": "parakeet_joint_decision_single_step.mlmodelc/analytics/coremldata.bin", "size": 243},
      {"type": "directory", "path": "parakeet_encoder.mlmodelc", "size": null},
      {"type": "file", "path": "parakeet_preprocessor.mlmodelc/weights/weight.bin", "size": 592384},
      {"type": "directory", "path": "parakeet_decoder.mlmodelc/weights", "size": null},
      {"type": "directory", "path": "parakeet_encoder.mlmodelc/weights", "size": null},
      {"type": "directory", "path": "parakeet_decoder.mlmodelc", "size": null},
      {"type": "directory", "path": "parakeet_joint_decision_single_step.mlmodelc/analytics", "size": null},
      {"type": "file", "path": "parakeet_joint_decision_single_step.mlmodelc/weights/weight.bin", "size": 3453388}
    ]
    """

    /// The 21 allowed runtime files, sorted ascending by relativePath — the
    /// canonical expected plan. Used to assert plan contents order-exactly.
    private static let expectedSortedRelativePaths: [String] = [
        "parakeet_decoder.mlmodelc/analytics/coremldata.bin",
        "parakeet_decoder.mlmodelc/coremldata.bin",
        "parakeet_decoder.mlmodelc/metadata.json",
        "parakeet_decoder.mlmodelc/model.mil",
        "parakeet_decoder.mlmodelc/weights/weight.bin",
        "parakeet_encoder.mlmodelc/analytics/coremldata.bin",
        "parakeet_encoder.mlmodelc/coremldata.bin",
        "parakeet_encoder.mlmodelc/metadata.json",
        "parakeet_encoder.mlmodelc/model.mil",
        "parakeet_encoder.mlmodelc/weights/weight.bin",
        "parakeet_joint_decision_single_step.mlmodelc/analytics/coremldata.bin",
        "parakeet_joint_decision_single_step.mlmodelc/coremldata.bin",
        "parakeet_joint_decision_single_step.mlmodelc/metadata.json",
        "parakeet_joint_decision_single_step.mlmodelc/model.mil",
        "parakeet_joint_decision_single_step.mlmodelc/weights/weight.bin",
        "parakeet_preprocessor.mlmodelc/analytics/coremldata.bin",
        "parakeet_preprocessor.mlmodelc/coremldata.bin",
        "parakeet_preprocessor.mlmodelc/metadata.json",
        "parakeet_preprocessor.mlmodelc/model.mil",
        "parakeet_preprocessor.mlmodelc/weights/weight.bin",
        "parakeet_vocab.json",
    ]

    /// Known runtime total: sum of the 21 filtered file sizes in the fixture.
    /// (decoder 14447282 + encoder 595217612 + joint_decision 3466823 +
    /// preprocessor 622386 + vocab 15519).
    private static let expectedTotalBytes: Int64 = 613_769_622

    private func decodedFixtureEntries() throws -> [ModelDownloader.TreeEntry] {
        let data = Data(Self.treeFixtureJSON.utf8)
        return try ModelDownloader.decodeTree(from: data)
    }

    // MARK: - a) Tree -> plan (allowlist + sort)

    func testTreeToPlanAppliesAllowlistAndSorts() throws {
        let entries = try decodedFixtureEntries()
        let plan = ModelDownloader.planFiles(from: entries)

        // Exactly the 21 runtime files survive the allowlist.
        XCTAssertEqual(plan.count, 21, "expected 21 filtered runtime files")

        let paths = plan.map(\.relativePath)

        // All four .mlmodelc model directories are represented.
        for dir in [
            "parakeet_preprocessor.mlmodelc",
            "parakeet_encoder.mlmodelc",
            "parakeet_decoder.mlmodelc",
            "parakeet_joint_decision_single_step.mlmodelc",
        ] {
            XCTAssertTrue(
                paths.contains { $0.hasPrefix(dir + "/") },
                "plan must include files from \(dir)"
            )
        }

        // The vocab is included.
        XCTAssertTrue(paths.contains("parakeet_vocab.json"), "vocab must be included")

        // `.gitattributes` (hidden) is NOT present.
        XCTAssertFalse(paths.contains(".gitattributes"), ".gitattributes must be excluded")

        // The synthetic stray (removed raw-logit joint) is NOT present — proves
        // the allowlist excludes future strays by default.
        XCTAssertFalse(
            paths.contains("parakeet_joint_logits_single_step.mlmodelc/weights/weight.bin"),
            "synthetic stray *_logits_* artifact must be excluded by the allowlist"
        )
        XCTAssertFalse(
            paths.contains { $0.hasPrefix("parakeet_joint_logits_single_step.mlmodelc/") },
            "no file from the stray logits model directory may appear"
        )

        // Result is sorted ascending by relativePath (both self-consistent and
        // equal to the canonical expected order).
        XCTAssertEqual(paths, paths.sorted(), "plan must be sorted ascending by relativePath")
        XCTAssertEqual(paths, Self.expectedSortedRelativePaths, "plan order must be canonical")
    }

    // MARK: - b) isAllowed unit cases

    func testIsAllowedTrueCases() {
        XCTAssertTrue(ModelDownloader.isAllowed(path: "parakeet_vocab.json"))
        XCTAssertTrue(
            ModelDownloader.isAllowed(path: "parakeet_encoder.mlmodelc/weights/weight.bin"))
    }

    func testIsAllowedFalseCases() {
        XCTAssertFalse(ModelDownloader.isAllowed(path: ".gitattributes"))
        XCTAssertFalse(ModelDownloader.isAllowed(path: "README.md"))
        XCTAssertFalse(ModelDownloader.isAllowed(path: ".something/x"))
        XCTAssertFalse(
            ModelDownloader.isAllowed(path: "parakeet_joint_logits_single_step.mlmodelc/model.mil"))
    }

    // MARK: - c) resolve URL construction

    func testDownloadURLForNestedPath() {
        let url = ModelDownloader.downloadURL(
            relativePath: "parakeet_encoder.mlmodelc/weights/weight.bin")
        XCTAssertEqual(
            url.absoluteString,
            "https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml/resolve/main/parakeet_encoder.mlmodelc/weights/weight.bin"
        )
    }

    func testDownloadURLForTopLevelFile() {
        let url = ModelDownloader.downloadURL(relativePath: "parakeet_vocab.json")
        XCTAssertEqual(
            url.absoluteString,
            "https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml/resolve/main/parakeet_vocab.json"
        )
    }

    // MARK: - d) total bytes

    func testTotalBytesEqualsSumOfPlan() throws {
        let entries = try decodedFixtureEntries()
        let plan = ModelDownloader.planFiles(from: entries)
        let total = plan.reduce(Int64(0)) { $0 + $1.size }
        XCTAssertEqual(
            total, Self.expectedTotalBytes,
            "sum of the 21 filtered file sizes must equal the known runtime total"
        )
    }

    // MARK: - e) LFS size preference
    //
    // The range-chunker sizes each file from the tree API. For an LFS file the
    // top-level `size` CAN be a small pointer size while the real size lives in
    // the nested `lfs.size`; if the planner used the pointer size the big
    // encoder weight would be chunked as one tiny GET and mis-downloaded. The
    // planner must prefer `lfs.size` when present. (For this repo the top-level
    // size already equals lfs.size — this guards against a repo that doesn't.)

    func testPlanPrefersLFSSizeOverTopLevelPointerSize() throws {
        let json = """
        [
          {"type": "file", "path": "parakeet_encoder.mlmodelc/weights/weight.bin",
           "size": 134, "lfs": {"size": 594211328}}
        ]
        """
        let entries = try ModelDownloader.decodeTree(from: Data(json.utf8))
        let plan = ModelDownloader.planFiles(from: entries)
        XCTAssertEqual(plan.count, 1)
        XCTAssertEqual(
            plan.first?.size, 594_211_328,
            "planner must use the resolved lfs.size, not the top-level pointer size")
    }

    func testPlanFallsBackToTopLevelSizeWhenNoLFS() throws {
        // Plain (non-LFS) file: no lfs object, so the top-level size is used.
        let json = """
        [ {"type": "file", "path": "parakeet_vocab.json", "size": 15519} ]
        """
        let entries = try ModelDownloader.decodeTree(from: Data(json.utf8))
        let plan = ModelDownloader.planFiles(from: entries)
        XCTAssertEqual(plan.first?.size, 15519)
    }

    // MARK: - f) chunk geometry

    func testChunkGeometrySmallFileIsSingleUnrangedSegment() {
        let chunk = ModelDownloader.chunkGeometry(100)
        XCTAssertEqual(chunk.count, 1)
        XCTAssertEqual(chunk.first, ModelDownloader.Chunk(offset: 0, length: 100, ranged: false))
    }

    func testChunkGeometryZeroSizeIsSingleEmptyUnrangedSegment() {
        let chunk = ModelDownloader.chunkGeometry(0)
        XCTAssertEqual(chunk, [ModelDownloader.Chunk(offset: 0, length: 0, ranged: false)])
    }

    func testChunkGeometryExactlyChunkSizeIsNotSplit() {
        // Boundary: a file of exactly chunkSize is a single un-ranged whole-file
        // GET (the split predicate is strictly `size > chunkSize`).
        let cs = ModelDownloader.chunkSize
        let chunk = ModelDownloader.chunkGeometry(cs)
        XCTAssertEqual(chunk, [ModelDownloader.Chunk(offset: 0, length: cs, ranged: false)])
    }

    func testChunkGeometryJustOverChunkSizeSplitsIntoTwo() {
        let cs = ModelDownloader.chunkSize
        let chunks = ModelDownloader.chunkGeometry(cs + 1)
        XCTAssertEqual(chunks, [
            ModelDownloader.Chunk(offset: 0, length: cs, ranged: true),
            ModelDownloader.Chunk(offset: cs, length: 1, ranged: true),
        ])
    }

    func testChunkGeometryEncoderWeightTilesWithoutGapsOrOverlap() {
        let cs = ModelDownloader.chunkSize
        let size: Int64 = 594_211_328          // the real encoder weight size
        let chunks = ModelDownloader.chunkGeometry(size)

        // ceil(size / chunkSize) chunks.
        let expectedCount = Int((size + cs - 1) / cs)
        XCTAssertEqual(chunks.count, expectedCount)
        XCTAssertGreaterThan(chunks.count, 1, "the encoder weight must be split")

        // Every chunk is ranged; all but the last are full-size.
        for (i, c) in chunks.enumerated() {
            XCTAssertTrue(c.ranged, "chunk \(i) of a large file must be ranged")
            if i < chunks.count - 1 {
                XCTAssertEqual(c.length, cs, "non-final chunk \(i) must be full chunkSize")
            } else {
                XCTAssertLessThanOrEqual(c.length, cs)
                XCTAssertGreaterThan(c.length, 0)
            }
        }

        // Contiguous cover of [0, size): each offset follows the previous chunk
        // exactly, and the lengths sum to the file size.
        var cursor: Int64 = 0
        for c in chunks {
            XCTAssertEqual(c.offset, cursor, "chunks must be contiguous with no gap/overlap")
            cursor += c.length
        }
        XCTAssertEqual(cursor, size, "chunks must cover exactly [0, size)")
    }

    // MARK: - g) bitmap-resume decision

    func testCanResumeRequiresMatchingBitmapAndPresentFile() {
        // Happy path: bitmap length matches the chunk count and the staging file
        // is still there -> safe to resume.
        XCTAssertTrue(
            ModelDownloader.canResume(savedBitmapCount: 36, expectedChunkCount: 36,
                                      destinationExists: true))
        // No saved bitmap -> fresh start.
        XCTAssertFalse(
            ModelDownloader.canResume(savedBitmapCount: nil, expectedChunkCount: 36,
                                      destinationExists: true))
        // Bitmap length disagrees with this file's chunking (e.g. chunkSize
        // changed, or the file size changed) -> distrust it, start fresh.
        XCTAssertFalse(
            ModelDownloader.canResume(savedBitmapCount: 12, expectedChunkCount: 36,
                                      destinationExists: true))
        // Bitmap matches but the staging file is gone -> the bits point at bytes
        // that aren't there, so start fresh.
        XCTAssertFalse(
            ModelDownloader.canResume(savedBitmapCount: 36, expectedChunkCount: 36,
                                      destinationExists: false))
    }

    // MARK: - h) repo-id parsing

    func testRepoIdFromFullTreeURL() {
        XCTAssertEqual(
            ModelDownloader.repoId(
                from: "https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml/tree/main"),
            "smdesai/parakeet-tdt-0.6b-v2-coreml")
    }

    func testRepoIdFromBareURL() {
        XCTAssertEqual(
            ModelDownloader.repoId(from: "https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml"),
            "smdesai/parakeet-tdt-0.6b-v2-coreml")
    }

    func testRepoIdFromBareOrgName() {
        XCTAssertEqual(
            ModelDownloader.repoId(from: "smdesai/parakeet-tdt-0.6b-v2-coreml"),
            "smdesai/parakeet-tdt-0.6b-v2-coreml")
    }

    func testRepoIdRejectsNonHuggingFace() {
        XCTAssertNil(ModelDownloader.repoId(from: "https://example.com/foo/bar"))
        XCTAssertNil(ModelDownloader.repoId(from: "not-a-repo"))
    }

    // MARK: - i) sentinel / isInstalled (non-destructive)
    //
    // rootDirectory()/isInstalled() resolve the REAL Application Support path
    // and rootDirectory() creates that directory as a side effect, so we do NOT
    // invoke them here — writing (or even touching) the real app-support root
    // from a unit test is out of bounds per the spec. We assert the install
    // constants directly, which is what isInstalled()/ensureInstalled() check
    // for. No filesystem is touched.

    func testSentinelNameIsComplete() {
        XCTAssertEqual(ModelDownloader.sentinelName, ".complete")
    }

    func testProgressDirNameIsHidden() {
        XCTAssertTrue(ModelDownloader.progressDirName.hasPrefix("."),
                      "the chunk-bitmap dir must be hidden so it isn't mistaken for a model file")
    }
}
