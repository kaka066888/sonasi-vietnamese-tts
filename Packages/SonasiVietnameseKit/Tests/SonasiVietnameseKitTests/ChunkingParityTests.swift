import Foundation
import Testing
@testable import SonasiVietnameseKit

private let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let goldenURL = root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden/chunk_golden.json")
private let dict = root.appendingPathComponent("third_party/sea-g2p/python/sea_g2p/sea_g2p.bin").path

private struct Golden: Decodable {
    struct Chunking: Decodable {
        let text: String, maxChars: Int, chunks: [String], gaps: [String], phonemes: [String], caps: [Int], syllables: [Int], cueOnly: [Bool]
    }
    struct Emotion: Decodable { let text: String, phonemes: String }
    struct Audio: Decodable { let segs: [[Double]], sr: Int, edge: [Int], bursts: Int }
    struct Pad: Decodable { let a: Int, b: Int, pause: Double, pad: Int, pad0: Int }
    struct Join: Decodable { let idx: [Int], silence: [Double], len: Int, sum: Double, abs_sum: Double }
    struct Babble: Decodable { let `case`: Int, ph: Int, cap: Int, res: [Int] }
    let chunking: [Chunking], emotion: [Emotion], audio: [Audio], pad: [Pad], join: Join, babble: [Babble], babblePhonemes: [String]
}

private func load() throws -> Golden { try JSONDecoder().decode(Golden.self, from: Data(contentsOf: goldenURL)) }

private func wave(_ segs: [[Double]]) -> [Float] {
    segs.flatMap { s in (0..<Int(s[0])).map { Float(s[1]) * ($0 % 2 == 0 ? 1 : -1) } }
}

@Test func chunkingPhonemesAndCapsMatchPython() throws {
    let g = try load(), g2p = try SeaG2P(dictionaryPath: dict)
    var chunks = 0
    for c in g.chunking {
        let r = try VieNeuChunker.chunks(for: c.text, g2p: g2p, maxChars: c.maxChars)
        #expect(r.chunks == c.chunks, "chunks lệch (maxChars \(c.maxChars)): \(c.text.prefix(50))\n got \(r.chunks)\n exp \(c.chunks)")
        #expect(r.gaps.map(\.rawValue) == c.gaps, "gaps lệch: \(c.text.prefix(50)) got \(r.gaps) exp \(c.gaps)")
        for (i, ch) in c.chunks.enumerated() {
            let ph = try VieNeuEmotion.phonemize(ch, g2p)
            #expect(ph == c.phonemes[i], "phoneme lệch: \(ch.prefix(40))\n got \(ph)\n exp \(c.phonemes[i])")
            #expect(VieNeuFrameCap.maxExpectedFrames(c.phonemes[i]) == c.caps[i], "cap: \(c.phonemes[i].prefix(30))")
            #expect(VieNeuFrameCap.syllableCount(c.phonemes[i]) == c.syllables[i], "syl: \(c.phonemes[i].prefix(30))")
            #expect(VieNeuFrameCap.isCueOnly(c.phonemes[i]) == c.cueOnly[i])
            chunks += 1
        }
    }
    #expect(chunks >= 90)
}

@Test func emotionPhonemizeMatchesPython() throws {
    let g = try load(), g2p = try SeaG2P(dictionaryPath: dict)
    for e in g.emotion {
        let ph = try VieNeuEmotion.phonemize(e.text, g2p)
        #expect(ph == e.phonemes, "\(e.text)\n got \(ph)\n exp \(e.phonemes)")
    }
}

@Test func audioUtilitiesMatchPython() throws {
    let g = try load()
    let waves = g.audio.map { wave($0.segs) }
    for (i, a) in g.audio.enumerated() {
        let e = VieNeuAudio.edgeSilence(waves[i], sr: a.sr)
        #expect([e.lead, e.tail] == a.edge, "edge case \(i): \(e) vs \(a.edge)")
        #expect(VieNeuAudio.countSpeechBursts(waves[i], sr: a.sr) == a.bursts, "bursts case \(i)")
    }
    for p in g.pad {
        #expect(VieNeuAudio.pausePadSamples(prev: waves[p.a], next: waves[p.b], sr: 8000, pauseSeconds: p.pause) == p.pad, "pad \(p.a)->\(p.b)")
        #expect(VieNeuAudio.pausePadSamples(prev: waves[p.a], next: waves[p.b], sr: 8000, pauseSeconds: 0) == p.pad0)
    }
    let joined = VieNeuAudio.join(g.join.idx.map { waves[$0] }, sr: 8000, silence: g.join.silence)
    #expect(joined.count == g.join.len)
    #expect(abs(joined.reduce(0.0) { $0 + Double($1) } - g.join.sum) < 1e-3)
    #expect(abs(joined.reduce(0.0) { $0 + Double(abs($1)) } - g.join.abs_sum) < 1e-3)
    for b in g.babble {
        let r = VieNeuAudio.babbleSuspect(waves[b.case], sr: 8000, phonemes: g.babblePhonemes[b.ph], capFrames: b.cap)
        #expect([r.suspect ? 1 : 0, r.syllables, r.bursts, r.frames] == b.res, "babble case \(b.case) ph \(b.ph): \(r) vs \(b.res)")
    }
}

@Test func synthesizerLongTextEndToEnd() throws {
    let model = root.appendingPathComponent("Models/onnx_int8")
    let synth = try VieNeuSynthesizer(modelDirectory: model, codecDirectory: root.appendingPathComponent("Models/codec"),
                                      dictionary: URL(fileURLWithPath: dict),
                                      voice: root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden/int8/voice.npz"))
    let text = "Cuộc họp bắt đầu lúc 9h30 tại tầng 3. Mọi người vui lòng đến đúng giờ [cười] nhé.\n\nDoanh thu quý 3 tăng 12% so với cùng kỳ, đạt 1,5 tỷ đồng. Chúng tôi sẽ deploy bản release mới lên server vào thứ Hai."
    var opts = VieNeuSynthesizer.Options(); opts.sampling = nil; opts.streaming = false   // greedy → tất định
    var streamed: [Float] = []
    let out = try synth.synthesize(text, options: opts, onAudio: { streamed += $0 })
    #expect(out.chunks.count >= 2)
    #expect(streamed == out.audio, "các phần streaming phải nối lại đúng bằng audio cuối")
    #expect(out.seconds > 5 && out.seconds < 40, "độ dài audio \(out.seconds) s")
    #expect(out.audio.contains { abs($0) > 0.05 })
    for c in out.chunks { #expect(c.frames <= c.capFrames, "vượt trần frame: \(c)") }
    let again = try synth.synthesize(text, options: opts)
    #expect(again.audio == out.audio, "greedy phải tất định")
    // streaming trong chunk: cùng nội dung chunk (greedy), khoảng nghỉ chỉ lệch rất nhẹ, các phần nối lại = audio cuối
    var sopts = opts; sopts.streaming = true
    var pieces: [[Float]] = []
    let streamedOut = try synth.synthesize(text, options: sopts, onAudio: { pieces.append($0) })
    #expect(pieces.flatMap { $0 } == streamedOut.audio)
    #expect(pieces.count > streamedOut.chunks.count, "chunk dài phải phát nhiều phần")
    #expect(abs(streamedOut.seconds - out.seconds) < 0.5, "streaming \(streamedOut.seconds) s vs \(out.seconds) s")
    // có lấy mẫu: vẫn chạy, ra audio hợp lý
    let sampled = try synth.synthesize(text, options: .init())
    #expect(sampled.seconds > 5 && sampled.seconds < 40)
    if ProcessInfo.processInfo.environment["EXPORT_WAVS"] != nil {
        let dir = root.appendingPathComponent("listen/swift_long"); try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, a) in [("greedy", out.audio), ("sampled", sampled.audio)] {
            var pcm = Data()
            for x in a { var v = Int16(max(-1, min(1, x)) * 32767).littleEndian; withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) } }
            func le<T: FixedWidthInteger>(_ x: T) -> Data { withUnsafeBytes(of: x.littleEndian) { Data($0) } }
            var h = Data("RIFF".utf8) + le(UInt32(36 + pcm.count)) + Data("WAVEfmt ".utf8) + le(UInt32(16)) + le(UInt16(1)) + le(UInt16(1))
            h += le(UInt32(48000)) + le(UInt32(96000)) + le(UInt16(2)) + le(UInt16(16)) + Data("data".utf8) + le(UInt32(pcm.count))
            try (h + pcm).write(to: dir.appendingPathComponent("\(name).wav"))
        }
    }
}

@Test func splitFirstShortensOnlyTheFirstChunk() throws {
    let g2p = try SeaG2P(dictionaryPath: dict)
    let text = "Xin chào quý vị, tôi xin báo cáo nhanh tình hình sản xuất quý ba. Nhà máy ở Bắc Ninh đang chạy ba ca, mỗi ca tám tiếng, sản lượng tăng 12% so với cùng kỳ năm ngoái, đạt 1,5 triệu sản phẩm. Tỉ lệ lỗi giảm còn 0,8 phần trăm."
    let base = try VieNeuChunker.chunks(for: text, g2p: g2p)
    let split = try VieNeuChunker.splitFirst(base, maxChars: 90)
    #expect(split.chunks.count > base.chunks.count)
    #expect(split.gaps.count == split.chunks.count - 1)
    #expect(split.chunks[0].unicodeScalars.count <= 100, "chunk đầu \(split.chunks[0].unicodeScalars.count) ký tự")
    func words(_ c: [String]) -> [String] { c.joined(separator: " ").split(separator: " ").map { $0.trimmingCharacters(in: .punctuationCharacters) } }
    #expect(words(split.chunks) == words(base.chunks), "không được mất/thêm từ")
    #expect(split.chunks.dropFirst(split.chunks.count - (base.chunks.count - 1)).elementsEqual(base.chunks.dropFirst()), "các chunk sau chunk đầu giữ nguyên")
    #expect(try VieNeuChunker.splitFirst(base, maxChars: 0) == base)
    let short = try VieNeuChunker.chunks(for: "Xin chào quý vị.", g2p: g2p)
    #expect(try VieNeuChunker.splitFirst(short, maxChars: 90) == short)
}
