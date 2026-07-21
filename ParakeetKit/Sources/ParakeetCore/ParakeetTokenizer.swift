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
public struct ParakeetTokenizer {
    let tokens: [Int: String]

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let raw = (try JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        var map: [Int: String] = [:]
        map.reserveCapacity(raw.count)
        for (key, value) in raw {
            if let id = Int(key), let piece = value as? String {
                map[id] = piece
            }
        }
        tokens = map
    }

    public func text(for ids: [Int]) -> String {
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
}
