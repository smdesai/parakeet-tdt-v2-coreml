import XCTest
@testable import parakeet_transcribe

final class KeywordBoosterTests: XCTestCase {
    // START: seq[0] of every keyword is always boosted, at any decode position.
    func testStartAlwaysBoostsFirstToken() {
        let b = KeywordBooster(keywordTokenSeqs: [[10, 11, 12]], alpha: 2.0)
        // No history yet.
        XCTAssertEqual(b.nextTokenBonuses(generated: [])[10], 2.0)
        // Arbitrary unrelated history still starts the keyword.
        let bonuses = b.nextTokenBonuses(generated: [99, 98, 97])
        XCTAssertEqual(bonuses[10], 2.0)
        // seq[1]/seq[2] are NOT armed here (no matching suffix).
        XCTAssertNil(bonuses[11])
        XCTAssertNil(bonuses[12])
    }

    // CONTINUE: seq[k] armed only when the last k generated == seq[0..<k].
    func testContinueArmsNextTokenOnPrefixMatch() {
        let b = KeywordBooster(keywordTokenSeqs: [[10, 11, 12]], alpha: 3.0)

        // After emitting [10], the last 1 == seq[0..<1] -> arm seq[1] == 11.
        let after10 = b.nextTokenBonuses(generated: [10])
        XCTAssertEqual(after10[11], 3.0)
        XCTAssertEqual(after10[10], 3.0)   // START still fires
        XCTAssertNil(after10[12])          // seq[2] not yet reachable

        // After [10, 11], last 2 == seq[0..<2] -> arm seq[2] == 12.
        let after1011 = b.nextTokenBonuses(generated: [10, 11])
        XCTAssertEqual(after1011[12], 3.0)
        XCTAssertEqual(after1011[10], 3.0) // START

        // A non-matching suffix must NOT arm the continuation.
        let mismatch = b.nextTokenBonuses(generated: [10, 99])
        XCTAssertNil(mismatch[12])
        XCTAssertNil(mismatch[11])         // last 1 == 99 != seq[0]
        XCTAssertEqual(mismatch[10], 3.0)  // START only
    }

    // CONTINUE respects the tail, not any interior match: [7,10] arms 11 (suffix 10),
    // and START (10) fires; 12 must not.
    func testContinueMatchesSuffixNotInterior() {
        let b = KeywordBooster(keywordTokenSeqs: [[10, 11, 12]], alpha: 1.0)
        let bonuses = b.nextTokenBonuses(generated: [7, 10])
        XCTAssertEqual(bonuses[11], 1.0)   // suffix [10] == seq[0..<1]
        XCTAssertEqual(bonuses[10], 1.0)   // START
        XCTAssertNil(bonuses[12])
    }

    // MAX-not-sum: a token reachable via two keywords gets alpha once, not 2*alpha.
    func testMaxNotSum() {
        // Both keywords START with token 10.
        let b = KeywordBooster(keywordTokenSeqs: [[10, 20], [10, 30]], alpha: 5.0)
        let bonuses = b.nextTokenBonuses(generated: [])
        XCTAssertEqual(bonuses[10], 5.0, "overlapping START must be max(alpha), not 2*alpha")

        // A token that is both a START and a CONTINUE of another keyword.
        // kw1 = [10, 40], kw2 = [40, 50]. After [10]: CONTINUE arms 40 (kw1),
        // START arms 40 (kw2). Still just alpha.
        let b2 = KeywordBooster(keywordTokenSeqs: [[10, 40], [40, 50]], alpha: 4.0)
        let bonuses2 = b2.nextTokenBonuses(generated: [10])
        XCTAssertEqual(bonuses2[40], 4.0, "START+CONTINUE on same token is still alpha")
    }

    // Empty booster => empty bonuses (baseline parity).
    func testEmptyBoosterYieldsNoBonuses() {
        let b = KeywordBooster(keywordTokenSeqs: [], alpha: 2.0)
        XCTAssertTrue(b.isEmpty)
        XCTAssertTrue(b.nextTokenBonuses(generated: []).isEmpty)
        XCTAssertTrue(b.nextTokenBonuses(generated: [1, 2, 3]).isEmpty)

        // All-empty sequences are filtered out -> also empty.
        let b2 = KeywordBooster(keywordTokenSeqs: [[], []], alpha: 2.0)
        XCTAssertTrue(b2.isEmpty)
        XCTAssertTrue(b2.nextTokenBonuses(generated: [5]).isEmpty)
    }

    // A single-token keyword is only ever a START (no continuation to arm).
    func testSingleTokenKeywordStartOnly() {
        let b = KeywordBooster(keywordTokenSeqs: [[42]], alpha: 2.0)
        let bonuses = b.nextTokenBonuses(generated: [42, 42])
        XCTAssertEqual(bonuses[42], 2.0)
        XCTAssertEqual(bonuses.count, 1)
    }
}
