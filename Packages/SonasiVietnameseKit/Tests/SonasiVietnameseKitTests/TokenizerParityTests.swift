import Foundation
import Testing
@testable import SonasiVietnameseKit

private let golden = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Golden")

@Test func tokenizerMatchesHuggingFace() throws {
    let tok = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: golden.appendingPathComponent("tokenizer.json")))
    struct Item: Decodable { let phonemes: String; let ids: [Int] }
    let items = try JSONDecoder().decode([Item].self, from: Data(contentsOf: golden.appendingPathComponent("tokenizer_corpus.json")))
    #expect(items.count >= 200)
    var bad = 0
    for it in items where tok.encode(it.phonemes) != it.ids {
        bad += 1
        if bad <= 5 { Issue.record("mismatch: \(it.phonemes.debugDescription)\n got \(tok.encode(it.phonemes))\n exp \(it.ids)") }
    }
    #expect(bad == 0)
}
