import Foundation

/// Maps TDT token ids to text. `parakeet_vocab.json` is a flat `{ "id": "piece" }`
/// map; pieces use the SentencePiece "▁" (U+2581) marker for a leading space.
/// Mirrors `ParakeetTokenizer` in parakeet-unified and `_tokens_to_text` in the
/// reference Python.
///
/// Only ids 0…1023 are real text. The blank id (1024) is never appended by the
/// decoder, and any id ≥ vocabSize is ignored here as a defensive guard — see
/// docs/swift-port-spec.md §1.2 (FluidAudio's dump appends spurious word-pieces at
/// 1024…1030, where 1024 == blank would otherwise render as "▁warming").
struct ParakeetTokenizer {
    let tokens: [Int: String]

    /// Reverse map piece -> id, built once in init. Used only by `encode` to arm
    /// the keyword booster; never touched on the decode/detokenize path. Only real
    /// text ids (0…vocabSize-1) are recorded, so `encode` can never emit blank.
    let pieceToId: [String: Int32]

    init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let raw = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        var map: [Int: String] = [:]
        map.reserveCapacity(raw.count)
        var rev: [String: Int32] = [:]
        rev.reserveCapacity(raw.count)
        for (key, value) in raw {
            if let id = Int(key), let piece = value as? String {
                map[id] = piece
                // Only real text ids can arm the booster; blank/non-text are dropped
                // exactly as `text(for:)` drops them. First writer wins on the rare
                // duplicate piece (ids are otherwise unique in a SentencePiece dump).
                if id >= 0, id < Const.vocabSize, !piece.isEmpty, rev[piece] == nil {
                    rev[piece] = Int32(id)
                }
            }
        }
        tokens = map
        pieceToId = rev
    }

    func text(for ids: [Int]) -> String {
        var pieces: [String] = []
        pieces.reserveCapacity(ids.count + 4)
        for id in ids {
            guard id >= 0, id < Const.vocabSize else { continue }   // drop blank / non-text
            guard let piece = tokens[id], !piece.isEmpty else { continue }
            if piece.hasPrefix("▁") {
                if !pieces.isEmpty { pieces.append(" ") }
                pieces.append(String(piece.dropFirst()))
            } else {
                pieces.append(piece)
            }
        }
        return pieces.joined().trimmingCharacters(in: .whitespaces)
    }

    /// Exact BPE encode of a keyword into the token-id sequence the model emits,
    /// used to arm the keyword booster. Returns ids in 0..<vocabSize only (never
    /// blank). Empty input → [].
    ///
    /// This tokenizer is **BPE** (verified against the real `.nemo` SentencePiece
    /// model: vocab 1024, model_type BPE). BPE is NOT greedy-longest-match — it
    /// starts from single characters and repeatedly merges the highest-priority
    /// adjacent pair. A greedy encoder diverges from `sp.encode()` on ~30% of
    /// domain words (e.g. "Ibrance", "pembrolizumab", "patient"), which would arm
    /// the wrong trie tokens and silently no-op the boost on exactly those words.
    ///
    /// Merge priority here needs **no extra asset**: in this model the per-piece
    /// merge score is the negative merge rank, so priority is monotonic with token
    /// id — a lower id was merged earlier and wins. So we merge the adjacent pair
    /// whose concatenation exists in `pieceToId` with the **lowest id**. This
    /// reproduces `sp.encode()` exactly (verified 40/40 on pharma + generic words;
    /// see EncodeTests and project memory `parakeet-bpe-encoder`).
    ///
    /// Word-start and interior spaces both map to the "▁" (U+2581) marker, matching
    /// SentencePiece's whitespace handling, so multi-word keyword phrases encode
    /// correctly too.
    func encode(_ word: String) -> [Int32] {
        let trimmed = word.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return [] }
        // SentencePiece replaces every space (including the word-initial one) with ▁.
        let normalized = "\u{2581}" + trimmed.replacingOccurrences(of: " ", with: "\u{2581}")

        // Start from single Unicode scalars (SentencePiece BPE operates on the
        // normalized character sequence). Each symbol is a vocab piece string.
        var syms: [String] = normalized.unicodeScalars.map { String($0) }

        // Repeatedly merge the best adjacent pair: the one whose concatenation is a
        // known piece with the SMALLEST id (== earliest BPE merge). Stop when no
        // adjacent pair is mergeable.
        while syms.count > 1 {
            var bestPos = -1
            var bestId = Int32.max
            for i in 0..<(syms.count - 1) {
                if let id = pieceToId[syms[i] + syms[i + 1]], id < bestId {
                    bestId = id
                    bestPos = i
                }
            }
            if bestPos < 0 { break }
            syms[bestPos] = syms[bestPos] + syms[bestPos + 1]
            syms.remove(at: bestPos + 1)
        }

        // Map final symbols to ids. A symbol with no piece entry is an unknown
        // character; drop it (it can only be a single unmergeable scalar, and the
        // booster has nothing to arm for it). pieceToId only holds ids in
        // 0..<vocabSize, so `encode` can never emit blank/non-text.
        var out: [Int32] = []
        out.reserveCapacity(syms.count)
        for s in syms {
            if let id = pieceToId[s], id >= 0, id < Int32(Const.vocabSize) {
                out.append(id)
            }
        }
        return out
    }
}
