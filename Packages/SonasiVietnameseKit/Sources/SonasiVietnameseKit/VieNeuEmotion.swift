import Foundation

/// Phoneme hóa giữ nguyên cue phi ngôn ngữ ([cười], [thở dài], [hắng giọng]) thành <|emotion_k|>,
/// port từ `phonemize_text_with_emotions` / `_normalize_sentence_keep_cues` của VieNeu.
enum VieNeuEmotion {
    private static let tagToK: [String: Int] = [
        "chuckle": 1, "cười": 1, "cuoi": 1,
        "sigh": 2, "thở dài": 2, "tho dai": 2,
        "clear throat": 3, "hắng giọng": 3, "hang giong": 3,
    ]
    private static let splitRE = try! NSRegularExpression(pattern: #"(\[[^\]]+\]|<\|emotion_\d+\|>)"#)
    private static let attachingPunct = Set(".,!?;:…)]}\"'’”")

    /// `[tag]`/`<|emotion_k|>` → `<|emotion_k|>`; nil nếu là cặp ngoặc không nhận ra.
    static func token(_ tag: String) -> String? {
        let t = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.hasPrefix("<|") { return t }
        let inner = String(t.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return tagToK[inner].map { "<|emotion_\($0)|>" }
    }

    static func hasCues(_ s: String) -> Bool { s.contains("[") || s.contains("<|emotion_") }

    /// Giống `re.split` với một nhóm bắt: phần lẻ (isTag) là tag.
    static func split(_ s: String) -> [(text: String, isTag: Bool)] {
        let ns = s as NSString
        var out: [(String, Bool)] = []
        var cursor = 0
        for m in splitRE.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            out.append((ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor)), false))
            out.append((ns.substring(with: m.range), true))
            cursor = m.range.location + m.range.length
        }
        out.append((ns.substring(from: cursor), false))
        return out
    }

    static func isBlank(_ s: String) -> Bool { s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    /// Normalize MỘT câu nhưng giữ cue thành <|emotion_k|>.
    static func normalizeKeepingCues(_ sentence: String, _ g2p: SeaG2P) throws -> String {
        guard hasCues(sentence) else { return try g2p.normalize(sentence, puncNorm: false) }
        var parts: [String] = []
        for (part, isTag) in split(sentence) {
            if isTag { parts.append(token(part) ?? part) }
            else if !isBlank(part) { parts.append(try g2p.normalize(part, puncNorm: false)) }
        }
        return parts.filter { !$0.isEmpty }.joined(separator: " ")
    }

    /// Phoneme hóa cả chunk (đường v3 Turbo): cue giữ thành token, dấu câu bám vào token trước.
    static func phonemize(_ text: String, _ g2p: SeaG2P) throws -> String {
        guard hasCues(text) else { return try g2p.phonemize(text, puncNorm: true) }
        var out = ""
        for (part, isTag) in split(text) {
            if isTag, let tok = token(part) {
                out = out.isEmpty ? tok : out + " " + tok
                continue
            }
            let ph = isBlank(part) ? "" : try g2p.phonemize(part, puncNorm: false)
            if ph.isEmpty { continue }
            if out.isEmpty { out = ph }
            else if let f = ph.first, attachingPunct.contains(f) { out += ph }
            else { out += " " + ph }
        }
        return try SeaG2P.puncNorm(out)
    }
}
