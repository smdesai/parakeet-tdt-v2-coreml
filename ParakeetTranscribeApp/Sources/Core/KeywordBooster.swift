import Foundation

/// Decode-time keyword booster: a token-trie shallow-fusion bias applied to the
/// joint's per-step token logits (spec: word-boost / docs/word-boost.md).
///
/// Each keyword is pre-encoded (by `ParakeetTokenizer.encode`) into a token-id
/// sequence. During TDT decode, before argmax at each frame, `nextTokenBonuses`
/// returns an additive bonus for the tokens that would advance a keyword match:
///
///   START:    the FIRST token of every keyword (`seq[0]`) is always boosted, so a
///             keyword can begin at any decode position.
///   CONTINUE: for every k in 1…seq.count-1, if the last k emitted tokens equal
///             `seq[0..<k]`, then `seq[k]` (the next token in that keyword) is
///             boosted — i.e. we're partway through the keyword and nudge the model
///             to finish it.
///
/// MAX-not-sum: when a token is reachable from several keywords (or as both a START
/// and a CONTINUE), it receives `alpha` ONCE (`max`), never a stacked `2*alpha`.
/// This keeps the bias bounded and predictable regardless of keyword-list overlap.
///
/// An empty booster (no keywords) returns `[:]` for every step, so the argmax path
/// falls back to the plain max and the decode is byte-identical to the baseline.
/// This is a bias, not a constraint: a strongly-preferred non-keyword token can
/// still win, so non-keyword transcription quality is preserved.
final class KeywordBooster {
    let alpha: Float
    private let keywordSeqs: [[Int32]]

    init(keywordTokenSeqs: [[Int32]], alpha: Float) {
        self.keywordSeqs = keywordTokenSeqs.filter { !$0.isEmpty }
        self.alpha = alpha
    }

    var isEmpty: Bool { keywordSeqs.isEmpty }

    /// Bonuses to add to the token logits for the next decode step, given the tokens
    /// emitted so far. See the type doc for START / CONTINUE / MAX-not-sum.
    func nextTokenBonuses(generated: [Int32]) -> [Int32: Float] {
        if keywordSeqs.isEmpty { return [:] }
        var bonuses: [Int32: Float] = [:]
        @inline(__always) func boost(_ t: Int32) {
            bonuses[t] = max(bonuses[t] ?? -.greatestFiniteMagnitude, alpha)   // MAX not sum
        }
        let n = generated.count
        for seq in keywordSeqs {
            boost(seq[0])   // START: a keyword may begin here.
            // CONTINUE: if the last k emitted == seq[0..<k], arm seq[k].
            let maxK = min(n, seq.count - 1)
            var k = 1
            while k <= maxK {
                var matches = true
                for j in 0..<k where generated[n - k + j] != seq[j] { matches = false; break }
                if matches { boost(seq[k]) }
                k += 1
            }
        }
        return bonuses
    }
}
