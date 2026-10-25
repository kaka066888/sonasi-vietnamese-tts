import Foundation
import SonasiVietnameseKit

/// Cô lập nguồn rò bộ nhớ: lặp riêng từng thành phần và in footprint.
func leakProbe(root: URL, precision: String, threads: Int) throws {
    let golden = root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden")
    let model = root.appendingPathComponent(precision == "int8" ? "Models/onnx_int8" : "Models/onnx_update")
    let tok = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: model.appendingPathComponent("tokenizer.json")))
    let cfg = try VieNeuEmbedder.Config(configJSON: Data(contentsOf: model.appendingPathComponent("config.json")))
    let emb = try VieNeuEmbedder(config: cfg, headsURL: model.appendingPathComponent("vieneu_v3_heads.npz"))
    let gen = try VieNeuGenerator(embedder: emb, modelDirectory: model, threads: threads)
    let codec = try VieNeuCodecStream(directory: root.appendingPathComponent("Models/codec"), threads: threads)
    let g2p = try SeaG2P(dictionaryPath: root.appendingPathComponent("third_party/sea-g2p/python/sea_g2p/sea_g2p.bin").path)
    let voice = try NPZ.read(golden.appendingPathComponent("\(precision)/voice.npz"))
    let ref = try voice["ref_codes"]!.ints()
    let refRows = stride(from: 0, to: ref.count, by: cfg.nVQ).map { Array(ref[$0..<($0 + cfg.nVQ)]) }
    let anchor = emb.speakerAnchor(try voice["speaker_emb"]!.floats())
    let text = "Hệ thống MES ghi nhận tỉ lệ lỗi giảm còn 0,8 phần trăm."
    func mb() -> Int { Int(VieNeuBenchmark.footprintMB()) }
    print("start \(mb()) MB")

    let pre = try VieNeuPrefill(modelURL: model.appendingPathComponent("vieneu_prefill.onnx"), threads: threads)
    let e0 = emb.embed(rows: emb.buildRows(phonemeIDs: tok.encode(try g2p.phonemize(text, puncNorm: true)), refCodes: refRows), anchor: anchor)
    let n0 = e0.count / cfg.hidden
    for i in 1...30 { _ = try pre.run(embeds: e0, length: n0); if i % 10 == 0 { print("prefill-only x\(i): \(mb()) MB") } }
    for i in 1...30 { try codec.reset(); if i % 10 == 0 { print("codec reset-only x\(i): \(mb()) MB") } }
    let fr0 = [[Int]](repeating: [Int](repeating: 5, count: 16), count: 4)
    try codec.reset()
    for i in 1...30 { _ = try codec.decode(fr0); if i % 10 == 0 { print("codec decode-only(no reset) x\(i): \(mb()) MB") } }
    for i in 1...200 { _ = try g2p.phonemize(text, puncNorm: true); if i % 100 == 0 { print("g2p x\(i): \(mb()) MB") } }
    let ph = try g2p.phonemize(text, puncNorm: true)
    let rows = emb.buildRows(phonemeIDs: tok.encode(ph), refCodes: refRows)

    var frames: [[Int]] = []
    for i in 1...30 {
        frames = try gen.generate(rows: rows, anchor: anchor, frameCap: 300).frames
        if i % 10 == 0 { print("generate x\(i): \(mb()) MB (\(frames.count) frames)") }
    }
    for i in 1...30 {
        try codec.reset()
        for s in stride(from: 0, to: frames.count, by: 4) { _ = try codec.decode(Array(frames[s..<min(s + 4, frames.count)])) }
        if i % 10 == 0 { print("codec x\(i): \(mb()) MB") }
    }
}
