import Foundation
import Testing
@testable import SonasiVietnameseKit

private let root = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let dict = root.appendingPathComponent("third_party/sea-g2p/python/sea_g2p/sea_g2p.bin").path
private let golden = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Golden")

@Test func seaG2PMatchesPythonPhonemes() throws {
    let g2p = try SeaG2P(dictionaryPath: dict)
    struct Item: Decodable { let text: String; let phonemes: String }
    let items = try JSONDecoder().decode([Item].self, from: Data(contentsOf: golden.appendingPathComponent("g2p_golden.json")))
    var checked = 0
    for it in items where !it.text.contains("[") {   // câu có emotion tag cần port riêng
        let got = try g2p.phonemize(it.text, puncNorm: true)
        #expect(got == it.phonemes, "\(it.text)\n got \(got)\n exp \(it.phonemes)")
        checked += 1
    }
    #expect(checked == 19)
}
