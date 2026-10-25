import Foundation

/// Byte-level BPE tokenizer khớp `tokenizer.json` của VieNeu-TTS v3 Turbo
/// (added-token split → NFC → regex split kiểu Qwen2 → byte→unicode → BPE theo merge-rank).
public struct VieNeuTokenizer: Sendable {
    public enum LoadError: Error { case malformed(String) }

    private static let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private let vocab: [String: Int]
    private let ranks: [String: Int]          // "a\u{0}b" -> rank
    private let addedTokens: [(content: String, id: Int)]  // dài trước
    private let unkID: Int
    private let byteToChar: [Character]

    public init(tokenizerJSON data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocabAny = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any]
        else { throw LoadError.malformed("model") }
        vocab = vocabAny
        var r = [String: Int](minimumCapacity: merges.count)
        for (i, m) in merges.enumerated() {
            if let p = m as? [String], p.count == 2 { r[p[0] + "\u{0}" + p[1]] = i }
            else if let s = m as? String {
                let p = s.split(separator: " ", maxSplits: 1).map(String.init)
                guard p.count == 2 else { throw LoadError.malformed("merge \(i)") }
                r[p[0] + "\u{0}" + p[1]] = i
            } else { throw LoadError.malformed("merge \(i)") }
        }
        ranks = r
        let added = (root["added_tokens"] as? [[String: Any]]) ?? []
        addedTokens = added.compactMap { d in
            guard let c = d["content"] as? String, let id = d["id"] as? Int else { return nil }
            return (c, id)
        }.sorted { $0.content.utf16.count > $1.content.utf16.count }
        unkID = added.first { ($0["content"] as? String) == "<|unk|>" }?["id"] as? Int ?? 0

        // Bảng byte → unicode của GPT-2.
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) { bs.append(b); cs.append(256 + n); n += 1 }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(UnicodeScalar(c)!) }
        byteToChar = table
    }

    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            // Tìm added token xuất hiện sớm nhất (ưu tiên token dài hơn khi cùng vị trí).
            var best: (range: Range<Substring.Index>, id: Int)?
            for t in addedTokens {
                if let r = rest.range(of: t.content),
                   best == nil || r.lowerBound < best!.range.lowerBound {
                    best = (r, t.id)
                }
            }
            guard let hit = best else { out += encodePlain(String(rest)); break }
            out += encodePlain(String(rest[..<hit.range.lowerBound]))
            out.append(hit.id)
            rest = rest[hit.range.upperBound...]
        }
        return out
    }

    private func encodePlain(_ s: String) -> [Int] {
        if s.isEmpty { return [] }
        let text = s.precomposedStringWithCanonicalMapping as NSString
        let re = try! NSRegularExpression(pattern: Self.pattern)
        var pieces: [String] = []
        var cursor = 0
        for m in re.matches(in: text as String, range: NSRange(location: 0, length: text.length)) {
            if m.range.location > cursor {
                pieces.append(text.substring(with: NSRange(location: cursor, length: m.range.location - cursor)))
            }
            pieces.append(text.substring(with: m.range))
            cursor = m.range.location + m.range.length
        }
        if cursor < text.length { pieces.append(text.substring(from: cursor)) }
        return pieces.flatMap(bpe)
    }

    private func bpe(_ piece: String) -> [Int] {
        var syms = piece.utf8.map { String(byteToChar[Int($0)]) }
        while syms.count > 1 {
            var bestRank = Int.max, bestIdx = -1
            for i in 0..<(syms.count - 1) {
                if let r = ranks[syms[i] + "\u{0}" + syms[i + 1]], r < bestRank { bestRank = r; bestIdx = i }
            }
            if bestIdx < 0 { break }
            syms.replaceSubrange(bestIdx...(bestIdx + 1), with: [syms[bestIdx] + syms[bestIdx + 1]])
        }
        return syms.map { vocab[$0] ?? unkID }
    }
}
