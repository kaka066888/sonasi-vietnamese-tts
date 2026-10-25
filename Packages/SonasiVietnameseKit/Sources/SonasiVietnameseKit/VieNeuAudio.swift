import Foundation

/// Đo im lặng, đếm cụm âm (babble guard) và nối audio các chunk — port từ `core_utils.py`.
public enum VieNeuAudio {
    public static let sampleRate = 48_000
    public static let gapSilence: [VieNeuChunker.Gap: Double] = [.para: 0.70, .sentence: 0.50, .minor: 0.30]

    public static func silence(for gaps: [VieNeuChunker.Gap]) -> [Double] { gaps.map { gapSilence[$0] ?? 0.50 } }

    /// `(lead, tail)` mẫu im lặng ở đầu/cuối (envelope mean|x| theo cửa sổ 10 ms, dưới -45 dB là im lặng).
    public static func edgeSilence(_ wav: [Float], sr: Int, threshDB: Float = -45, winSeconds: Float = 0.01) -> (lead: Int, tail: Int) {
        let n = wav.count, win = max(1, Int(winSeconds * Float(sr))), nWin = n / win
        if nWin == 0 { return (n, 0) }
        let thresh = Float(pow(10.0, Double(threshDB) / 20))
        var first = -1, last = -1
        for w in 0..<nWin {
            var s: Float = 0
            for i in (w * win)..<((w + 1) * win) { s += abs(wav[i]) }
            if s / Float(win) > thresh { if first < 0 { first = w }; last = w }
        }
        if first < 0 { return (n, 0) }
        return (first * win, n - (last + 1) * win)
    }

    /// Số mẫu zeros chèn giữa hai chunk để khoảng nghỉ thật (đuôi trước + zeros + đầu sau) đạt `pauseSeconds`.
    public static func pausePadSamples(prev: [Float], next: [Float], sr: Int, pauseSeconds: Double) -> Int {
        var (leadPrev, tail) = edgeSilence(prev, sr: sr)
        if leadPrev == prev.count { tail = prev.count }
        let lead = edgeSilence(next, sr: sr).lead
        return max(0, Int(pauseSeconds * Double(sr)) - tail - lead)
    }

    /// Ghép chunk theo `silence[i]` = khoảng nghỉ tối thiểu giữa chunk i và i+1; audio từng chunk giữ nguyên.
    public static func join(_ chunks: [[Float]], sr: Int, silence: [Double]) -> [Float] {
        guard var out = chunks.first else { return [] }
        for i in 1..<max(1, chunks.count) {
            let pad = pausePadSamples(prev: chunks[i - 1], next: chunks[i], sr: sr, pauseSeconds: i - 1 < silence.count ? silence[i - 1] : 0)
            if pad > 0 { out += [Float](repeating: 0, count: pad) }
            out += chunks[i]
        }
        return out
    }

    // MARK: babble guard

    static let babbleMaxSyllables = 3, burstThreshDB: Float = -18, burstMinGapMS = 60, burstMinMS = 30

    /// Số cụm năng lượng (xấp xỉ số tiếng) trong waveform mono.
    public static func countSpeechBursts(_ wav: [Float], sr: Int) -> Int {
        let hop = max(1, Int(Float(sr) * 0.010)), n = wav.count / hop
        if n == 0 { return 0 }
        var env = [Float](repeating: 0, count: n)
        for k in 0..<n {
            var s: Float = 0
            for i in (k * hop)..<((k + 1) * hop) { s += wav[i] * wav[i] }
            env[k] = (s / Float(hop)).squareRoot()
        }
        guard let peak = env.max(), peak > 1e-6 else { return 0 }
        let thresh = peak * Float(pow(10.0, Double(burstThreshDB) / 20))
        let minGap = max(1, burstMinGapMS / 10), minLen = max(1, burstMinMS / 10)
        var bursts: [(Int, Int)] = [], start: Int?, lastOn = 0
        for (i, e) in env.enumerated() where e > thresh {
            if start == nil { start = i }
            else if i - lastOn > minGap { bursts.append((start!, lastOn)); start = i }
            lastOn = i
        }
        if let s = start { bursts.append((s, lastOn)) }
        return bursts.filter { $0.1 - $0.0 + 1 >= minLen }.count
    }

    public struct Babble: Equatable, Sendable { public var suspect: Bool, syllables: Int, bursts: Int, frames: Int }

    /// Nghi "nói thêm" cho chunk ≤ 3 tiếng: nhiều cụm âm hơn số tiếng, hoặc chunk ≤ 2 tiếng chạy sát trần frame.
    public static func babbleSuspect(_ wav: [Float], sr: Int, phonemes: String, capFrames: Int, frames: Int? = nil) -> Babble {
        let n = frames ?? Int((Double(wav.count) / (Double(sr) / 12.5)).rounded())
        if VieNeuFrameCap.isCueOnly(phonemes) { return Babble(suspect: n >= capFrames - 1, syllables: 0, bursts: 0, frames: n) }
        let syl = VieNeuFrameCap.syllableCount(phonemes)
        if syl == 0 || syl > babbleMaxSyllables || phonemes.contains("<|emotion_") { return Babble(suspect: false, syllables: syl, bursts: 0, frames: 0) }
        let bursts = countSpeechBursts(wav, sr: sr)
        let hitCap = syl <= 2 && n >= capFrames - 1
        return Babble(suspect: bursts > syl || hitCap, syllables: syl, bursts: bursts, frames: n)
    }

    /// Bản sinh lại `new` có đáng thay `old` không.
    public static func babblePrefer(_ new: Babble, over old: Babble) -> Bool {
        if new.suspect != old.suspect { return !new.suspect }
        return new.bursts < old.bursts || (new.bursts == old.bursts && new.frames < old.frames)
    }
}
