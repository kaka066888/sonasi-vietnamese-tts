import Foundation
import Testing
@testable import SonasiVietnameseKit

private let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private func dirs(_ precision: String) -> (model: URL, golden: URL) {
    (root.appendingPathComponent(precision == "int8" ? "Models/onnx_int8" : "Models/onnx_update"),
     root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden/\(precision)"))
}

private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).map { abs($0 - $1) }.max() ?? 0
}

/// fp32: phải khớp chặt (chứng minh pipeline đúng). int8: lượng tử hóa động nên chỉ đòi cosine cao;
/// Python int8 vs Python fp32 cũng chỉ đạt cosine ~0,995.
@Test(arguments: ["fp32", "int8"]) func embeddingAndPrefillMatchGolden(precision: String) throws {
    let (model, golden) = dirs(precision)
    let tokenizer = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: model.appendingPathComponent("tokenizer.json")))
    let cfg = try VieNeuEmbedder.Config(configJSON: Data(contentsOf: model.appendingPathComponent("config.json")))
    let emb = try VieNeuEmbedder(config: cfg, headsURL: model.appendingPathComponent("vieneu_v3_heads.npz"))
    let prefill = try VieNeuPrefill(modelURL: model.appendingPathComponent("vieneu_prefill.onnx"))

    let voice = try NPZ.read(golden.appendingPathComponent("voice.npz"))
    let speaker = try voice["speaker_emb"]!.floats()
    let ref = try voice["ref_codes"]!.ints()
    let refRows = stride(from: 0, to: ref.count, by: cfg.nVQ).map { Array(ref[$0..<($0 + cfg.nVQ)]) }
    let anchorGolden = try voice["anchor"]!.floats()
    let anchor = emb.speakerAnchor(speaker)
    #expect(maxAbsDiff(anchor, anchorGolden) < 1e-4, "anchor diff \(maxAbsDiff(anchor, anchorGolden))")

    struct Item: Decodable { let phonemes: String }
    let items = try JSONDecoder().decode([Item].self, from: Data(contentsOf: golden.deletingLastPathComponent().appendingPathComponent("g2p_golden.json")))

    var worstEmb: Float = 0, worstHidden: Float = 0, worstCos: Float = 1, worstRel: Float = 0
    for (i, it) in items.enumerated() {
        let g = try NPZ.read(golden.appendingPathComponent(String(format: "s%02d.npz", i)))
        let ids = tokenizer.encode(it.phonemes)
        #expect(ids == (try g["token_ids"]!.ints()), "tokens s\(i)")

        let rows = emb.buildRows(phonemeIDs: ids, refCodes: refRows)
        let gRows = try g["rows"]!.ints()
        #expect(rows.flatMap { $0 } == gRows, "rows s\(i)")

        let e = emb.embed(rows: rows, anchor: anchor)
        let h = cfg.hidden
        let head = try g["embeds_head"]!.floats(), tail = try g["embeds_tail"]!.floats()
        let dEmb = max(maxAbsDiff(Array(e[0..<(4 * h)]), head), maxAbsDiff(Array(e[(e.count - 2 * h)...]), tail))
        worstEmb = max(worstEmb, dEmb)
        let sum = e.reduce(0.0) { $0 + Double($1) }
        let gSum = try g["embeds_sum"]!.doubles()[0]
        #expect(abs(sum - gSum) < max(1e-2, abs(gSum) * 1e-4), "sum s\(i): \(sum) vs \(gSum)")

        let out = try prefill.run(embeds: e, length: rows.count)
        let gh = try g["prefill_last_hidden"]!.floats()
        let dH = maxAbsDiff(out.lastHidden, gh)
        worstHidden = max(worstHidden, dH)
        let dot = zip(out.lastHidden, gh).reduce(0) { $0 + $1.0 * $1.1 }
        let na = out.lastHidden.reduce(0) { $0 + $1 * $1 }.squareRoot(), nb = gh.reduce(0) { $0 + $1 * $1 }.squareRoot()
        worstCos = min(worstCos, dot / (na * nb))
        let ref = gh.map(abs).max()!
        worstRel = max(worstRel, dH / ref)
        #expect(out.pastK.count == 12 && out.pastV.count == 12)
    }
    print("[\(precision)] worst embeds diff \(worstEmb), prefill hidden: max abs \(worstHidden), max rel \(worstRel), min cosine \(worstCos)")
    #expect(worstEmb < 1e-4)
    if precision == "fp32" { #expect(worstHidden < 2e-3) } else { #expect(worstCos > 0.99) }
}
