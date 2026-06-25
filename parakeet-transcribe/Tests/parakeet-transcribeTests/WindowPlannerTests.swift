import XCTest
@testable import parakeet_transcribe

final class WindowPlannerTests: XCTestCase {
    let sr = Const.sampleRate
    let planner = WindowPlanner()   // 15s window, 2.5s ctx, 10s center

    private func check(_ n: Int, file: StaticString = #filePath, line: UInt = #line) {
        let specs = planner.plan(nSamples: n)
        XCTAssertFalse(specs.isEmpty, "no windows for n=\(n)", file: file, line: line)

        // Centers tile [0, n) contiguously, no gaps, no overlap.
        XCTAssertEqual(specs.first!.centerLo, 0, "first center must start at 0", file: file, line: line)
        XCTAssertEqual(specs.last!.centerHi, n, "last center must end at n", file: file, line: line)
        for i in 1..<max(1, specs.count) where specs.count > 1 {
            XCTAssertEqual(specs[i].centerLo, specs[i - 1].centerHi,
                           "centers must be contiguous at \(i)", file: file, line: line)
        }

        for s in specs {
            // No left padding: window start clamps to 0; trueLen never exceeds window.
            XCTAssertGreaterThanOrEqual(s.winStart, 0, file: file, line: line)
            XCTAssertLessThanOrEqual(s.trueLen, planner.winSamples, file: file, line: line)
            // True length never reads past the audio.
            XCTAssertLessThanOrEqual(s.winStart + s.trueLen, n, file: file, line: line)
            // Emitted center lies within the window's true samples.
            XCTAssertGreaterThanOrEqual(s.centerLo, s.winStart, file: file, line: line)
            XCTAssertLessThanOrEqual(s.centerHi, s.winStart + s.trueLen, file: file, line: line)
        }
    }

    func testShortAudioSingleWindow() {
        let n = 1 * sr
        let specs = planner.plan(nSamples: n)
        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0], WindowSpec(winStart: 0, trueLen: n, centerLo: 0, centerHi: n))
    }

    func testExactlyOneWindow() {
        let n = 15 * sr                       // == winSamples
        let specs = planner.plan(nSamples: n)
        XCTAssertEqual(specs.count, 1, "exactly 15s should be a single full window")
        check(n)
    }

    func testJustOverOneWindow() {
        let n = 15 * sr + 1                    // 15.0001s -> must split
        let specs = planner.plan(nSamples: n)
        XCTAssertGreaterThan(specs.count, 1)
        XCTAssertEqual(specs[0].winStart, 0, "first window has no left context")
        check(n)
    }

    func test31s()  { check(31 * sr) }
    func test66s()  { check(66 * sr) }

    func testFirstNoLeftLastNoRight() {
        let n = 66 * sr
        let specs = planner.plan(nSamples: n)
        XCTAssertEqual(specs.first!.winStart, 0, "first window starts at 0 (no left ctx)")
        // last window reads to the end of audio (no right ctx beyond n)
        let last = specs.last!
        XCTAssertEqual(last.winStart + last.trueLen, n, "last window ends at n (no right ctx)")
    }

    func testCenterWidthIs10s() {
        XCTAssertEqual(planner.centerSamples, 10 * sr)
    }
}
