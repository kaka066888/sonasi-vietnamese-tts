import Foundation

/// Trần số frame hợp lý cho một chunk theo phoneme + đếm âm tiết (port `core_utils.max_expected_frames`,
/// `syllable_count`, `is_cue_only`).
public enum VieNeuFrameCap {
    static let maxFramesPerPhone = 2.0, slack = 24
    static let singleWordMaxFrames = 13, perExtraSyllable = 5, maxSyllablesForCap = 4, singleWordMaxPhones = 24

    private static let markupRE = try! NSRegularExpression(pattern: #"<\|emotion_\d+\|>|</?en>"#)
    private static let ipaVowels: Set<Unicode.Scalar> = Set("aeiouyæɐɑɒɔəɘɛɜɤɯɵøœʉʊʌɪɨɚɝᵻᵿ".unicodeScalars)

    static func stripMarkup(_ s: String) -> String {
        markupRE.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: (s as NSString).length), withTemplate: "")
    }

    private static func isAlpha(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: true
        default: false
        }
    }
    private static func isDigit(_ u: Unicode.Scalar) -> Bool { u.properties.generalCategory == .decimalNumber }

    public static func isCueOnly(_ phonemes: String) -> Bool {
        phonemes.contains("<|emotion_") && !stripMarkup(phonemes).unicodeScalars.contains(where: isAlpha)
    }

    public static func syllableCount(_ phonemes: String) -> Int {
        var total = 0
        for tok in stripMarkup(phonemes).split(whereSeparator: { $0.isWhitespace }) {
            var groups = 0, inV = false, consonantSeen = true, alpha = false
            for ch in tok.unicodeScalars {
                if isAlpha(ch) { alpha = true }
                if ipaVowels.contains(ch) {
                    if !inV && consonantSeen { groups += 1 }
                    inV = true; consonantSeen = false
                } else if ch == "ː" || ch == "ˈ" || ch == "ˌ" || isDigit(ch) {
                    if (ch == "ˈ" || ch == "ˌ") && groups > 0 { inV = false; consonantSeen = true } else { inV = false }
                } else {
                    inV = false; consonantSeen = true
                }
            }
            if alpha { total += max(1, groups) }
        }
        return total
    }

    public static func maxExpectedFrames(_ phonemes: String) -> Int {
        let effLen = stripMarkup(phonemes).unicodeScalars.count
        var cap = slack + Int((maxFramesPerPhone * Double(effLen)).rounded(.up))
        if isCueOnly(phonemes) { return min(cap, singleWordMaxFrames) }
        if !phonemes.contains("<|emotion_") {
            let syl = max(1, syllableCount(phonemes))
            if syl <= maxSyllablesForCap && effLen <= singleWordMaxPhones * syl {
                cap = min(cap, singleWordMaxFrames + perExtraSyllable * (syl - 1))
            }
        }
        return cap
    }
}
