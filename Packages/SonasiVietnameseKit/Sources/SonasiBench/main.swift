import Foundation
import SonasiVietnameseKit

// swift run -c release SonasiBench <repo-root> <int8|fp32> <threads> [sustainedSeconds]
let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: SonasiBench <repo-root> <int8|fp32> <threads> [sustainedSeconds]"); exit(2) }
let root = URL(fileURLWithPath: args[1]), precision = args[2], threads = Int(args[3])!
if args.count > 4 && args[4] == "leak" { try leakProbe(root: root, precision: precision, threads: threads); exit(0) }
if args.count > 4 && (args[4] == "long" || args[4] == "longloop" || args[4] == "export") {
    struct LongText: Decodable { let name: String; let text: String }
    let texts = try JSONDecoder().decode([LongText].self, from: Data(contentsOf: root.appendingPathComponent("App/SonasiBench/Resources/long_texts.json")))
    let synth = try VieNeuSynthesizer(
        modelDirectory: root.appendingPathComponent(precision == "int8" ? "Models/onnx_int8" : "Models/onnx_update"),
        codecDirectory: root.appendingPathComponent("Models/codec"),
        dictionary: root.appendingPathComponent("third_party/sea-g2p/python/sea_g2p/sea_g2p.bin"),
        voice: root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden/\(precision)/voice.npz"), threads: threads,
        prefillThreads: args.count > 5 ? Int(args[5]) : nil)
    _ = try synth.synthesize("Xin chào.")   // warm-up
    if args[4] == "export" {
        let dir = root.appendingPathComponent("listen/first_chunk"); try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, firstMax) in [("A_chunk_dau_ngan_90", 90), ("B_chunk_dau_dai_nhu_cu", 0)] {
            var o = VieNeuSynthesizer.Options(); o.firstChunkMaxChars = firstMax
            o.sampling = .init(seed: 42)                       // cùng seed để so sánh công bằng
            let out = try synth.synthesize(texts[1].text, options: o)
            var pcm = Data()
            for x in out.audio { var v = Int16(max(-1, min(1, x)) * 32767).littleEndian; withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) } }
            func le<T: FixedWidthInteger>(_ x: T) -> Data { withUnsafeBytes(of: x.littleEndian) { Data($0) } }
            var h = Data("RIFF".utf8) + le(UInt32(36 + pcm.count)) + Data("WAVEfmt ".utf8) + le(UInt32(16)) + le(UInt16(1)) + le(UInt16(1))
            h += le(UInt32(48000)) + le(UInt32(96000)) + le(UInt16(2)) + le(UInt16(16)) + Data("data".utf8) + le(UInt32(pcm.count))
            try (h + pcm).write(to: dir.appendingPathComponent("\(name).wav"))
            print("\(name): \(out.chunks.count) chunk, \(String(format: "%.1f", out.seconds)) s; chunk đầu: \(out.chunks.prefix(4).map { $0.text })")
        }
        exit(0)
    }
    if args[4] == "longloop" {
        let item = texts[1]
        for i in 1...25 {
            let r = try VieNeuLongBenchmark.run(synth, text: item.text, log: { _ in })
            print(String(format: "#%2d RTF %.2f footprint %d MB (now %d MB)", i, r.rtf, Int(r.peakFootprintMB), Int(VieNeuBenchmark.footprintMB())))
        }
        exit(0)
    }
    for t in texts {
        print("== \(t.name), \(precision), \(threads) luồng ==")
        let tp = DispatchTime.now().uptimeNanoseconds
        let plan = try synth.plan(t.text)
        print(String(format: "lập kế hoạch chunk (%d chunk): %.0f ms", plan.chunks.count, Double(DispatchTime.now().uptimeNanoseconds - tp) / 1e6))
        let r = try VieNeuLongBenchmark.run(synth, text: t.text, log: { _ in })
        VieNeuLongBenchmark.format(r).forEach { print($0) }
    }
    exit(0)
}
let golden = root.appendingPathComponent("Packages/SonasiVietnameseKit/Tests/Golden")
struct Item: Decodable { let text: String }
let texts = try JSONDecoder().decode([Item].self, from: Data(contentsOf: golden.appendingPathComponent("g2p_golden.json"))).map(\.text).filter { !$0.contains("[") }
let res = VieNeuBenchmark.Resources(
    modelDirectory: root.appendingPathComponent(precision == "int8" ? "Models/onnx_int8" : "Models/onnx_update"),
    codecDirectory: root.appendingPathComponent("Models/codec"),
    dictionary: root.appendingPathComponent("third_party/sea-g2p/python/sea_g2p/sea_g2p.bin"),
    voice: golden.appendingPathComponent("\(precision)/voice.npz"), texts: texts)
let bench = try VieNeuBenchmark(res, threads: threads)
try bench.run { print($0) }
if args.count > 4 { try bench.sustained(seconds: Double(args[4])!) { print($0) } }
