import Foundation
import Testing
@testable import SonasiVietnameseKit

private let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let goldenRoot = root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden")
private let codecDir = root.appendingPathComponent("Models/codec")

private struct Rig {
    let model: URL, golden: URL
    let tokenizer: VieNeuTokenizer
    let cfg: VieNeuEmbedder.Config
    let embedder: VieNeuEmbedder
    let generator: VieNeuGenerator
    let refRows: [[Int]]
    let anchor: [Float]
    let phonemes: [String]

    init(_ precision: String) throws {
        model = root.appendingPathComponent(precision == "int8" ? "Models/onnx_int8" : "Models/onnx_update")
        golden = goldenRoot.appendingPathComponent(precision)
        tokenizer = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: model.appendingPathComponent("tokenizer.json")))
        cfg = try VieNeuEmbedder.Config(configJSON: Data(contentsOf: model.appendingPathComponent("config.json")))
        embedder = try VieNeuEmbedder(config: cfg, headsURL: model.appendingPathComponent("vieneu_v3_heads.npz"))
        generator = try VieNeuGenerator(embedder: embedder, modelDirectory: model)
        let voice = try NPZ.read(golden.appendingPathComponent("voice.npz"))
        let ref = try voice["ref_codes"]!.ints()
        let nVQ = cfg.nVQ
        refRows = stride(from: 0, to: ref.count, by: nVQ).map { Array(ref[$0..<($0 + nVQ)]) }
        anchor = embedder.speakerAnchor(try voice["speaker_emb"]!.floats())
        struct Item: Decodable { let phonemes: String }
        phonemes = try JSONDecoder().decode([Item].self, from: Data(contentsOf: goldenRoot.appendingPathComponent("g2p_golden.json"))).map(\.phonemes)
    }

    func rows(_ i: Int) -> [[Int]] { embedder.buildRows(phonemeIDs: tokenizer.encode(phonemes[i]), refCodes: refRows) }
}

/// fp32: backbone khớp ~1e-6 nhưng graph acoustic là int8 động nên mã codebook có thể rẽ nhánh khác Python
/// (xem ORT.session). Ngưỡng hồi quy đo ngày 2026-10-24: 13/20 câu trùng từng mã; câu lệch vẫn chung ≥ 4 frame đầu,
/// kết thúc bằng eos, độ dài ±15%.
@Test func fp32FramesCloseToGolden() throws {
    let rig = try Rig("fp32")
    var identical = 0
    for i in 0..<20 {
        let g = try NPZ.read(rig.golden.appendingPathComponent(String(format: "s%02d.npz", i)))
        let gFrames = try g["frames"]!.ints()
        let r = try rig.generator.generate(rows: rig.rows(i), anchor: rig.anchor, frameCap: try g["frame_cap"]!.ints()[0])
        let flat = r.frames.flatMap { $0 }
        if flat == gFrames { identical += 1; continue }
        let common = zip(flat, gFrames).prefix { $0 == $1 }.count / 16
        #expect(common >= 4, "s\(i): chỉ chung \(common) frame đầu")
        #expect(r.eosAt != nil, "s\(i) không gặp eos")
        let ratio = Double(r.frames.count) / Double(gFrames.count / 16)
        #expect(abs(ratio - 1) <= 0.15, "s\(i): \(r.frames.count) frame vs \(gFrames.count / 16)")
    }
    #expect(identical >= 12, "chỉ \(identical)/20 câu trùng golden")
}

/// int8: không đòi trùng bit (nhiễu lượng tử hóa) — chỉ cần kết thúc bằng eos, độ dài cùng cỡ golden.
@Test func int8GenerationTerminatesSensibly() throws {
    let rig = try Rig("int8")
    for i in [0, 1, 5, 14] {
        let g = try NPZ.read(rig.golden.appendingPathComponent(String(format: "s%02d.npz", i)))
        let gCount = try g["frames"]!.ints().count / 16
        let r = try rig.generator.generate(rows: rig.rows(i), anchor: rig.anchor, frameCap: try g["frame_cap"]!.ints()[0])
        #expect(r.eosAt != nil, "s\(i) không gặp eos")
        let ratio = Double(r.frames.count) / Double(gCount)
        #expect(ratio > 0.7 && ratio < 1.4, "s\(i): \(r.frames.count) vs golden \(gCount)")
    }
}

/// Codec decode_step theo nhóm frame phải ra đúng độ dài và dạng sóng của decode_full trong golden.
@Test(arguments: [1, 4, 1000]) func codecStreamMatchesGolden(group: Int) throws {
    let codec = try VieNeuCodecStream(directory: codecDir)
    for i in [0, 6, 14] {
        let g = try NPZ.read(goldenRoot.appendingPathComponent(String(format: "fp32/s%02d.npz", i)))
        let f = try g["frames"]!.ints()
        let frames = stride(from: 0, to: f.count, by: 16).map { Array(f[$0..<($0 + 16)]) }
        try codec.reset()
        var wav: [Float] = []
        for s in stride(from: 0, to: frames.count, by: group) {
            wav += try codec.decode(Array(frames[s..<min(s + group, frames.count)]))
        }
        let len = try g["wav_len"]!.ints()[0]
        #expect(wav.count == len, "s\(i) group \(group): \(wav.count) vs \(len)")
        let head = try g["wav_head"]!.floats()
        let n = min(head.count, wav.count)
        let d = zip(wav.prefix(n), head.prefix(n)).map { abs($0 - $1) }.max() ?? 0
        let rms = (wav.reduce(0.0) { $0 + Double($1 * $1) } / Double(wav.count)).squareRoot()
        print("codec s\(i) group \(group): len \(wav.count), head max diff \(d), rms \(rms) vs \(try g["wav_rms"]!.doubles()[0])")
        #expect(d < 2e-3)
    }
}





/// Xuất wav end-to-end bằng pipeline Swift (đặt EXPORT_WAVS=1) để nghe thử.
@Test(.enabled(if: ProcessInfo.processInfo.environment["EXPORT_WAVS"] != nil)) func exportSwiftWavs() throws {
    let rig = try Rig("fp32")
    let outDir = root.appendingPathComponent("listen/swift_fp32")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let codec = try VieNeuCodecStream(directory: codecDir)
    for i in [0, 14, 17] {
        let g = try NPZ.read(rig.golden.appendingPathComponent(String(format: "s%02d.npz", i)))
        let r = try rig.generator.generate(rows: rig.rows(i), anchor: rig.anchor, frameCap: try g["frame_cap"]!.ints()[0])
        try codec.reset()
        var wav: [Float] = []
        for s in stride(from: 0, to: r.frames.count, by: 4) { wav += try codec.decode(Array(r.frames[s..<min(s + 4, r.frames.count)])) }
        var pcm = Data()
        for x in wav { var v = Int16(max(-1, min(1, x)) * 32767).littleEndian; withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) } }
        func le<T: FixedWidthInteger>(_ x: T) -> Data { withUnsafeBytes(of: x.littleEndian) { Data($0) } }
        var h = Data("RIFF".utf8) + le(UInt32(36 + pcm.count)) + Data("WAVEfmt ".utf8) + le(UInt32(16)) + le(UInt16(1)) + le(UInt16(1))
        h += le(UInt32(48000)) + le(UInt32(96000)) + le(UInt16(2)) + le(UInt16(16)) + Data("data".utf8) + le(UInt32(pcm.count))
        try (h + pcm).write(to: outDir.appendingPathComponent(String(format: "s%02d.wav", i)))
    }
}
