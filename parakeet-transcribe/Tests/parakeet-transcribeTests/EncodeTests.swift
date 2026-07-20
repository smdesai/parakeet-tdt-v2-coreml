import XCTest
@testable import parakeet_transcribe

/// Exercises `ParakeetTokenizer.encode` against the real vocab. No CoreML needed —
/// just the JSON vocab. If the vocab can't be found (e.g. a checkout without the
/// model artifacts), the tests skip gracefully rather than fail.
final class EncodeTests: XCTestCase {
    /// Locate `parakeet_vocab.json`, searching the two known repo locations relative
    /// to this test file, then any env override. Returns nil if not found.
    private func vocabURL() -> URL? {
        if let p = ProcessInfo.processInfo.environment["PARAKEET_VOCAB"],
           FileManager.default.fileExists(atPath: p) {
            return URL(fileURLWithPath: p)
        }
        // .../parakeet-transcribe/Tests/parakeet-transcribeTests/EncodeTests.swift
        // repo root is three directories up from this file's directory.
        var root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<3 { root.deleteLastPathComponent() }
        let candidates = [
            root.appendingPathComponent("parakeet_coreml_v2_final/parakeet_vocab.json"),
            root.appendingPathComponent("ParakeetTranscribeApp/Resources/Models/parakeet_vocab.json"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private func loadTokenizer() throws -> ParakeetTokenizer {
        guard let url = vocabURL() else {
            throw XCTSkip("parakeet_vocab.json not found; skipping encode round-trip tests")
        }
        return try ParakeetTokenizer(contentsOf: url)
    }

    // Common words encode to non-empty, in-range ids and round-trip via text(for:).
    func testEncodeRoundTrip() throws {
        let tok = try loadTokenizer()
        for word in ["hello", "world", "the", "parakeet", "transcribe", "boosting"] {
            let ids = tok.encode(word)
            XCTAssertFalse(ids.isEmpty, "encode(\(word)) should not be empty")
            for id in ids {
                XCTAssertGreaterThanOrEqual(id, 0, "id \(id) for \(word) must be >= 0")
                XCTAssertLessThan(id, Int32(Const.vocabSize), "id \(id) for \(word) must be < vocabSize (never blank)")
            }
            // Round-trip: detokenizing the encoded ids reproduces the word
            // (lowercased/space-normalized — encode is word-initial so the first
            // piece carries the leading-space marker).
            let text = tok.text(for: ids.map { Int($0) })
            XCTAssertEqual(text.lowercased(), word.lowercased(),
                           "round-trip mismatch: \(word) -> \(ids) -> \(text)")
        }
    }

    // EXACT BPE ground truth: encode() must reproduce the token ids that the real
    // SentencePiece model (`sp.encode()`) emits — NOT a greedy longest-match
    // approximation (which diverges on ~30% of these). The expected ids below were
    // generated from the tokenizer inside parakeet-tdt-0.6b-v2.nemo. This is the
    // regression guard against silently reverting to greedy segmentation. See
    // project memory `parakeet-bpe-encoder`.
    func testEncodeExactBPEGroundTruth() throws {
        let tok = try loadTokenizer()
        let cases: [(String, [Int32])] = [
            ("Talzenna",      [65, 50, 864, 22, 449]),
            ("Xeljanz",       [819, 890, 211, 849, 25, 864]),
            ("Ibrance",       [34, 840, 828, 410]),        // greedy gets this WRONG
            ("pembrolizumab", [24, 217, 840, 78, 829, 325, 192, 187]), // greedy WRONG
            ("osimertinib",   [8, 826, 95, 611, 4, 654]),
            ("patient",       [24, 10, 825, 68]),          // greedy gets this WRONG
            ("metformin",     [816, 514, 4]),
            ("Keytruda",      [457, 820, 833, 821, 828, 287, 823]),
            ("cena",          [16, 22, 823]),
            ("Sinha",         [71, 4, 827, 823]),
        ]
        for (word, expected) in cases {
            XCTAssertEqual(tok.encode(word), expected,
                           "BPE encode mismatch for \(word) — did encode() regress to greedy?")
        }
    }

    // A multi-word phrase: interior spaces map to ▁ so each word starts a new
    // leading-space piece; round-trips to the same phrase.
    func testEncodePhraseRoundTrip() throws {
        let tok = try loadTokenizer()
        let ids = tok.encode("the patient")
        XCTAssertFalse(ids.isEmpty)
        XCTAssertEqual(tok.text(for: ids.map { Int($0) }).lowercased(), "the patient")
    }

    // Empty / whitespace-only keyword encodes to nothing (so it's dropped upstream).
    func testEncodeEmptyIsEmpty() throws {
        let tok = try loadTokenizer()
        XCTAssertTrue(tok.encode("").isEmpty)
        XCTAssertTrue(tok.encode("   ").isEmpty)
    }

    // The first emitted id must be a word-initial (leading-space) piece.
    func testEncodeStartsWithLeadingSpacePiece() throws {
        let tok = try loadTokenizer()
        let ids = tok.encode("hello")
        guard let first = ids.first, let piece = tok.tokens[Int(first)] else {
            return XCTFail("encode(hello) produced no first piece")
        }
        XCTAssertTrue(piece.hasPrefix("\u{2581}"),
                      "first piece \(piece) should carry the word-initial ▁ marker")
    }
}
