import Foundation

/// Chia văn bản dài thành chunk cho đường v3 Turbo, port từ `core_utils.py` + `phonemize_text.py` của VieNeu
/// (`normalize_to_chunks_v3_with_gaps`). Mọi độ dài tính theo mã Unicode (như `len()` của Python).
public enum VieNeuChunker {
    public enum Gap: String, Sendable { case para, sentence, minor }

    public struct Result: Sendable, Equatable {
        public var chunks: [String]
        public var gaps: [Gap]          // gaps[i] giữa chunks[i] và chunks[i+1]
    }

    // MARK: public API

    public static func chunks(for text: String, g2p: SeaG2P, maxChars: Int = 256, minChunkChars: Int = 20) throws -> Result {
        guard !text.isEmpty else { return Result(chunks: [], gaps: []) }
        let keepCues = VieNeuEmotion.hasCues(text)
        var chunks: [String] = [], gaps: [Gap] = []
        for para in splitNewlines(text) where !blank(para) {
            let sentences = splitIntoSentences(para)
            if sentences.isEmpty { continue }
            let normalized = try sentences.map { keepCues ? try VieNeuEmotion.normalizeKeepingCues($0, g2p) : try g2p.normalize($0, puncNorm: false) }
            let paraChunks = packSentences(normalized, maxChars: maxChars)
            if paraChunks.isEmpty { continue }
            if !chunks.isEmpty { gaps.append(.para) }
            for (j, ch) in paraChunks.enumerated() {
                if j > 0 { gaps.append(.sentence) }
                chunks.append(ch)
            }
        }
        chunks = try chunks.map { try SeaG2P.puncNorm($0) }
        gaps = gaps.enumerated().map { $1 == .para ? .para : classifyGap(chunks[$0]) }
        return mergeShort(chunks, gaps, minChars: minChunkChars)
    }

    /// Cắt chunk ĐẦU thành các mảnh ≤ `maxChars` ở ranh giới câu/dấu phẩy để prefill (prompt ngắn hơn) xong sớm → TTFA thấp.
    /// Không cắt nếu mảnh đầu sẽ ngắn hơn `minFirstChars` (tránh chunk đầu 1–2 từ dễ "nói thêm").
    public static func splitFirst(_ plan: Result, maxChars: Int, minFirstChars: Int = 20) throws -> Result {
        guard maxChars > 0, let first = plan.chunks.first, plen(first) > maxChars else { return plan }
        let pieces = try packSentences(splitIntoSentences(first), maxChars: maxChars).map { try SeaG2P.puncNorm($0) }
        guard pieces.count > 1, effectiveLen(pieces[0]) >= minFirstChars else { return plan }
        var chunks = pieces + plan.chunks.dropFirst()
        var gaps = pieces.dropLast().map { classifyGap($0) }
        gaps.append(plan.gaps.first ?? .sentence)
        gaps += plan.gaps.dropFirst()
        if plan.chunks.count == 1 { gaps.removeLast() }       // chunk duy nhất: không có gap sau mảnh cuối
        chunks = chunks.map { $0 }
        return Result(chunks: chunks, gaps: gaps)
    }

    // MARK: helpers (string)

    static func plen(_ s: String) -> Int { s.unicodeScalars.count }
    static func blank(_ s: String) -> Bool { s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    static func strip(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }

    private static let newlineRE = try! NSRegularExpression(pattern: "[\r\n]+")
    static func splitNewlines(_ s: String) -> [String] { split(s, by: newlineRE) }

    /// `re.split` không có nhóm bắt.
    static func split(_ s: String, by re: NSRegularExpression) -> [String] {
        let ns = s as NSString
        var out: [String] = [], cursor = 0
        for m in re.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            out.append(ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor)))
            cursor = m.range.location + m.range.length
        }
        out.append(ns.substring(from: cursor))
        return out
    }

    // MARK: sentence splitting

    private static let openToClose: [Unicode.Scalar: Unicode.Scalar] = [
        "(": ")", "[": "]", "{": "}", "\u{201C}": "\u{201D}", "\u{2018}": "\u{2019}", "«": "»", "‹": "›", "「": "」", "『": "』",
    ]
    private static let openers = Set(openToClose.keys), closers = Set(openToClose.values)
    private static let sentEnd: Set<Unicode.Scalar> = [".", "!", "?", "…"]
    private static let trailingClose: Set<Unicode.Scalar> = closers.union(["\"", "'", "\u{2019}", "\u{201D}"])

    private static func isSpace(_ u: Unicode.Scalar) -> Bool { u.properties.isWhitespace }

    private static func scan(_ text: String, quoteAware: Bool) -> (sentences: [String], balanced: Bool) {
        let s = Array(text.unicodeScalars), n = s.count
        func str(_ a: Int, _ b: Int) -> String { var v = String.UnicodeScalarView(); v.append(contentsOf: s[a..<b]); return String(v) }
        var sentences: [String] = []
        var start = 0, i = 0, depth = 0, inQuote = false
        while i < n {
            let ch = s[i]
            if quoteAware && ch == "\"" { inQuote.toggle() }
            else if quoteAware && openers.contains(ch) { depth += 1 }
            else if quoteAware && closers.contains(ch) { if depth > 0 { depth -= 1 } }
            else if sentEnd.contains(ch) && depth == 0 && !inQuote {
                var j = i + 1
                while j < n && sentEnd.contains(s[j]) { j += 1 }
                while j < n && trailingClose.contains(s[j]) { j += 1 }
                if j >= n || isSpace(s[j]) {
                    sentences.append(str(start, j)); start = j; i = j; continue
                }
                i = j; continue
            }
            i += 1
        }
        if start < n { sentences.append(str(start, n)) }
        return (sentences.map(strip).filter { !$0.isEmpty }, depth == 0 && !inQuote)
    }

    static func splitIntoSentences(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        let (sentences, balanced) = scan(text, quoteAware: true)
        return balanced ? sentences : scan(text, quoteAware: false).sentences
    }

    // MARK: packing

    private static let connWords: Set<String> = ["và", "nhưng", "hoặc", "song", "rồi", "nên", "vì", "nếu", "khi", "để", "do", "bởi"]
    private static let connPairs: Set<String> = Set([
        ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("mỗi", "khi"), ("đến", "khi"), ("tới", "khi"), ("cho", "nên"),
        ("cho", "đến"), ("bởi", "vì"), ("nếu", "như"), ("tuy", "nhiên"), ("thế", "nhưng"), ("vì", "vậy"), ("vì", "thế"),
        ("do", "đó"), ("sau", "đó"),
    ].map { "\($0.0)\u{0}\($0.1)" })
    private static let connStrip = CharacterSet(charactersIn: "\"'\u{201C}\u{201D}\u{2018}\u{2019}()[]«»…")
    private static let numberWords: Set<String> = Set(
        "không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ linh lẻ phẩy chấm".split(separator: " ").map(String.init))
    private static let tokenRE = try! NSRegularExpression(pattern: #"<en>.*?</en>|\S+"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    private static let minorPunctRE = try! NSRegularExpression(pattern: #"(?<=[,;:\-–—])\s+"#)

    private static func key(_ t: String) -> String { t.trimmingCharacters(in: connStrip).lowercased() }
    private static func pair(_ a: String, _ b: String) -> Bool { connPairs.contains("\(a)\u{0}\(b)") }
    private static func isNumberWord(_ t: String) -> Bool {
        let k = key(t)
        return numberWords.contains(k) || (!k.isEmpty && k.unicodeScalars.allSatisfy { $0.properties.generalCategory == .decimalNumber })
    }

    private static func tailSlack(_ maxChars: Int) -> Int { min(15, maxChars / 8) }
    private static func fits(_ cur: Int, _ add: Int, _ maxChars: Int) -> Bool {
        let total = cur > 0 ? cur + 1 + add : add
        let slack = tailSlack(maxChars)
        return total <= maxChars || (add <= slack && total <= maxChars + slack)
    }

    private static func tokens(_ s: String) -> [String] {
        let ns = s as NSString
        return tokenRE.matches(in: s, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }

    private static func spanLen(_ w: [String], _ a: Int, _ b: Int) -> Int {
        b <= a ? 0 : w[a..<b].reduce(0) { $0 + plen($1) } + (b - a - 1)
    }

    private static func balancedCut(_ w: [String], _ start: Int, _ target: Double, _ maxChars: Int, _ minLeft: Int) -> Int {
        var bestNat: (Double, Int)?, bestPlain: (Double, Int)?
        var endCap = start + 1
        var j = start + 1
        while j < w.count {
            let left = spanLen(w, start, j)
            if left > maxChars { break }
            endCap = j
            let k = key(w[j]), prev = key(w[j - 1]), nxt = j + 1 < w.count ? key(w[j + 1]) : ""
            let d = abs(Double(left) - target)
            let inside = pair(prev, k)
            let natural = (connWords.contains(k) || pair(k, nxt)) && !connWords.contains(prev)
            if natural && !inside && left >= minLeft && (bestNat == nil || d < bestNat!.0) { bestNat = (d, j) }
            let plainOK = !inside && !(isNumberWord(w[j - 1]) && isNumberWord(w[j]))
            if plainOK && (bestPlain == nil || d < bestPlain!.0) { bestPlain = (d, j) }
            j += 1
        }
        if let b = bestNat { return b.1 }
        if let b = bestPlain { return b.1 }
        var end = endCap
        while end > start + 1 && end < w.count && pair(key(w[end - 1]), key(w[end])) { end -= 1 }
        return end
    }

    private static func splitLongPart(_ part: String, _ maxChars: Int) -> [String] {
        let words = tokens(part), minLeft = maxChars / 3
        var pieces: [String] = [], start = 0
        while start < words.count {
            let rest = spanLen(words, start, words.count)
            if rest <= maxChars { pieces.append(words[start...].joined(separator: " ")); break }
            var full = start + 1
            while full < words.count && spanLen(words, start, full + 1) <= maxChars { full += 1 }
            if fits(spanLen(words, start, full), spanLen(words, full, words.count), maxChars) {
                pieces.append(words[start...].joined(separator: " ")); break
            }
            let nPieces = (rest + maxChars - 1) / maxChars
            let end = balancedCut(words, start, Double(rest) / Double(nPieces), maxChars, minLeft)
            pieces.append(words[start..<end].joined(separator: " "))
            start = end
        }
        return pieces
    }

    static func packSentences(_ sentences: [String], maxChars: Int) -> [String] {
        var out: [String] = [], buffer = ""
        for raw in sentences {
            let sentence = strip(raw)
            if sentence.isEmpty { continue }
            if plen(sentence) > maxChars {
                if !buffer.isEmpty { out.append(buffer); buffer = "" }
                for rawPart in split(sentence, by: minorPunctRE) {
                    let part = strip(rawPart)
                    if part.isEmpty { continue }
                    if fits(plen(buffer), plen(part), maxChars) {
                        buffer = buffer.isEmpty ? part : buffer + " " + part
                    } else {
                        if !buffer.isEmpty { out.append(buffer) }
                        buffer = part
                        if plen(buffer) > maxChars {
                            let pieces = splitLongPart(buffer, maxChars)
                            out.append(contentsOf: pieces.dropLast())
                            buffer = pieces.last ?? ""
                        }
                    }
                }
            } else if !buffer.isEmpty && !fits(plen(buffer), plen(sentence), maxChars) {
                out.append(buffer); buffer = sentence
            } else {
                buffer = buffer.isEmpty ? sentence : buffer + " " + sentence
            }
        }
        if !buffer.isEmpty { out.append(buffer) }
        return out.map(strip).filter { !$0.isEmpty }
    }

    // MARK: gaps + merge short

    static func classifyGap(_ chunk: String) -> Gap {
        let c = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = c.unicodeScalars.last else { return .minor }
        return ".!?".unicodeScalars.contains(last) ? .sentence : .minor
    }

    private static let emotionTokenRE = try! NSRegularExpression(pattern: #"<\|emotion_\d+\|>"#)
    static func effectiveLen(_ chunk: String) -> Int {
        let ns = chunk as NSString
        let stripped = emotionTokenRE.stringByReplacingMatches(in: chunk, range: NSRange(location: 0, length: ns.length), withTemplate: "")
        return plen(strip(stripped))
    }

    static func mergeShort(_ chunksIn: [String], _ gapsIn: [Gap], minChars: Int) -> Result {
        var chunks = chunksIn, gaps = gapsIn
        while chunks.count > 1 {
            let short = chunks.indices.filter { effectiveLen(chunks[$0]) < minChars }
            if short.isEmpty { break }
            let i = short.min { effectiveLen(chunks[$0]) < effectiveLen(chunks[$1]) || (effectiveLen(chunks[$0]) == effectiveLen(chunks[$1]) && $0 < $1) }!
            // Python: max(sides, key=(ưu_tiên, -len)) — hòa thì lấy "R" (đứng trước trong danh sách).
            var sides: [(Bool, Int, Character)] = []
            if i < chunks.count - 1 { sides.append((gaps[i] != .para, -plen(chunks[i + 1]), "R")) }
            if i > 0 { sides.append((gaps[i - 1] != .para, -plen(chunks[i - 1]), "L")) }
            var best = sides[0]
            for s in sides.dropFirst() where (s.0 ? 1 : 0, s.1) > (best.0 ? 1 : 0, best.1) { best = s }
            if best.2 == "R" {
                chunks[i] = chunks[i] + " " + chunks.remove(at: i + 1); gaps.remove(at: i)
            } else {
                chunks[i - 1] = chunks[i - 1] + " " + chunks.remove(at: i); gaps.remove(at: i - 1)
            }
        }
        return Result(chunks: chunks, gaps: gaps)
    }
}
