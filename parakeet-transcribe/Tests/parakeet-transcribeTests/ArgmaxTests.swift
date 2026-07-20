import CoreML
import XCTest
@testable import parakeet_transcribe

final class ArgmaxTests: XCTestCase {
    /// Build a tiny fp32 [1,1,1,count] logit vector.
    private func logits(_ values: [Float]) -> MLMultiArray {
        MLArray.float(values, shape: [1, 1, 1, values.count], dataType: .float32)
    }

    // Empty bonuses => plain argmax, identical to a raw max.
    func testEmptyBonusesEqualsPlainArgmax() {
        let vals: [Float] = [0.1, 0.5, 0.2, 0.9, 0.3]
        let arr = logits(vals)
        let idx = MLArray.argmax(logits: arr, count: vals.count, bonuses: [:])
        XCTAssertEqual(idx, 3, "plain argmax should pick index 3 (0.9)")
    }

    // A bonus large enough flips the winner.
    func testBonusFlipsWinner() {
        let vals: [Float] = [0.1, 0.5, 0.2, 0.9, 0.3]
        let arr = logits(vals)
        // Plain winner is 3 (0.9). Boost id 1 (0.5) by +1.0 -> 1.5 > 0.9.
        let idx = MLArray.argmax(logits: arr, count: vals.count, bonuses: [1: 1.0])
        XCTAssertEqual(idx, 1, "a large enough bonus must flip the winner to id 1")
    }

    // A too-small bonus does NOT flip the winner.
    func testSmallBonusDoesNotFlip() {
        let vals: [Float] = [0.1, 0.5, 0.2, 0.9, 0.3]
        let arr = logits(vals)
        // Boost id 1 (0.5) by +0.1 -> 0.6 < 0.9; winner stays 3.
        let idx = MLArray.argmax(logits: arr, count: vals.count, bonuses: [1: 0.1])
        XCTAssertEqual(idx, 3, "a bonus too small to overtake the max must not flip")
    }

    // The blank index (count-1) is selectable both as plain max and when boosted.
    func testBlankIndexSelectable() {
        // Mirror the real width: 1025 token logits, blank at index 1024.
        let count = Const.vocabSize + 1
        var vals = [Float](repeating: 0.0, count: count)
        vals[Const.blankId] = 5.0           // blank is the plain max
        let arr = logits(vals)
        XCTAssertEqual(MLArray.argmax(logits: arr, count: count, bonuses: [:]), Const.blankId,
                       "blank at index \(Const.blankId) must be a selectable plain max")

        // A boost on the blank id is honored (index is in range).
        var vals2 = [Float](repeating: 0.0, count: count)
        vals2[3] = 1.0
        let arr2 = logits(vals2)
        let idx = MLArray.argmax(logits: arr2, count: count, bonuses: [Int32(Const.blankId): 2.0])
        XCTAssertEqual(idx, Const.blankId, "a boosted blank id (in range) is selectable")
    }

    // Out-of-range bonus ids are ignored (negative and >= count).
    func testOutOfRangeBonusIgnored() {
        let vals: [Float] = [0.1, 0.5, 0.2, 0.9, 0.3]
        let arr = logits(vals)
        let idx = MLArray.argmax(logits: arr, count: vals.count,
                                 bonuses: [-1: 100.0, 5: 100.0, 999: 100.0])
        XCTAssertEqual(idx, 3, "bonuses on out-of-range ids must be ignored")
    }

    // `count` scans only the first `count` logits even if the array is wider.
    func testCountLimitsScan() {
        // 6 values but scan only the first 4; the max (index 5) is out of scope.
        let vals: [Float] = [0.1, 0.5, 0.2, 0.4, 9.0, 9.9]
        let arr = logits(vals)
        let idx = MLArray.argmax(logits: arr, count: 4, bonuses: [:])
        XCTAssertEqual(idx, 1, "only the first 4 logits are scanned; index 1 (0.5) wins")
    }
}
