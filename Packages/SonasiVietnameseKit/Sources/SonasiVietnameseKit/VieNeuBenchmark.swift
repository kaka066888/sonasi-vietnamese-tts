import Foundation

/// Benchmark TTFA / RTF / RAM / thermalState dùng chung cho CLI (Mac) và app iOS.
public final class VieNeuBenchmark: @unchecked Sendable {
    public struct Resources: Sendable {
        public var modelDirectory: URL      // config.json, tokenizer.json, vieneu_*.onnx, heads npz
        public var codecDirectory: URL      // moss_audio_tokenizer_decode_step.onnx (+ .data), codec_browser_onnx_meta.json
        public var dictionary: URL          // sea_g2p.bin
        public var voice: URL               // voice.npz (speaker_emb, ref_codes)
        public var texts: [String]
        public init(modelDirectory: URL, codecDirectory: URL, dictionary: URL, voice: URL, texts: [String]) {
            self.modelDirectory = modelDirectory; self.codecDirectory = codecDirectory
            self.dictionary = dictionary; self.voice = voice; self.texts = texts
        }
    }

    struct Run { var g2p = 0.0, prefill = 0.0, ttfa = 0.0, gen = 0.0, total = 0.0, audio = 0.0; var frames = 0, promptRows = 0 }

    public static let firstGroup = 2, group = 4

    private let res: Resources
    private let threads: Int
    private let tokenizer: VieNeuTokenizer
    private let embedder: VieNeuEmbedder
    private let generator: VieNeuGenerator
    private let codec: VieNeuCodecStream
    private let g2p: SeaG2P
    private let refRows: [[Int]]
    private let anchor: [Float]
    public let loadSummary: String

    public init(_ res: Resources, threads: Int) throws {
        self.res = res; self.threads = threads
        let m = res.modelDirectory
        let rss0 = Self.footprintMB(); var t0 = Self.now()
        tokenizer = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: m.appendingPathComponent("tokenizer.json")))
        let cfg = try VieNeuEmbedder.Config(configJSON: Data(contentsOf: m.appendingPathComponent("config.json")))
        embedder = try VieNeuEmbedder(config: cfg, headsURL: m.appendingPathComponent("vieneu_v3_heads.npz"))
        let tHeads = Self.now() - t0; t0 = Self.now()
        generator = try VieNeuGenerator(embedder: embedder, modelDirectory: m, threads: threads)
        codec = try VieNeuCodecStream(directory: res.codecDirectory, threads: threads)
        g2p = try SeaG2P(dictionaryPath: res.dictionary.path)
        let tSessions = Self.now() - t0
        let voice = try NPZ.read(res.voice)
        let ref = try voice["ref_codes"]!.ints(), n = cfg.nVQ
        refRows = stride(from: 0, to: ref.count, by: n).map { Array(ref[$0..<($0 + n)]) }
        anchor = embedder.speakerAnchor(try voice["speaker_emb"]!.floats())
        loadSummary = "load: heads+tokenizer \(Self.ms(tHeads)), sessions+codec+g2p \(Self.ms(tSessions)), footprint +\(Int(Self.footprintMB() - rss0)) MB (total \(Int(Self.footprintMB())) MB)"
    }

    /// Chạy toàn bộ `texts` một lượt (có 1 lượt warm-up).
    public func run(log: (String) -> Void) throws {
        log("thermal \(Self.thermal()), threads \(threads)")
        log(loadSummary)
        _ = try synth(res.texts[min(5, res.texts.count - 1)])
        var runs: [Run] = []
        for t in res.texts { runs.append(try synth(t)) }
        func avg(_ f: (Run) -> Double) -> Double { runs.map(f).reduce(0, +) / Double(runs.count) }
        let audio = runs.map(\.audio).reduce(0, +), total = runs.map(\.total).reduce(0, +)
        let frames = runs.map(\.frames).reduce(0, +), gen = runs.map(\.gen).reduce(0, +)
        log(String(format: "câu %d, audio %.1f s, compute %.1f s → RTF %.2f (<1 = nhanh hơn thời gian thực)", runs.count, audio, total, total / audio))
        log("TTFA (\(Self.firstGroup) frame đầu ≈ \(Self.firstGroup * 80) ms audio): mean \(Self.ms(avg(\.ttfa))), min \(Self.ms(runs.map(\.ttfa).min()!)), max \(Self.ms(runs.map(\.ttfa).max()!))")
        log("  g2p \(Self.ms(avg(\.g2p))), prefill \(Self.ms(avg(\.prefill))) (prompt ~\(Int(avg { Double($0.promptRows) })) hàng), sinh \(Self.ms(avg(\.gen)))")
        let perFrame = gen / Double(frames)
        log(String(format: "mỗi frame %.1f ms / 80 ms audio → %.2fx thời gian thực; footprint %d MB", perFrame * 1000, 0.08 / perFrame, Int(Self.footprintMB())))
        log("thermal sau chạy: \(Self.thermal())")
    }

    /// Chạy liên tục `seconds` giây, in nhiệt + RTF mỗi 15 s. `shouldStop` để hủy.
    public func sustained(seconds: Double, shouldStop: () -> Bool = { false }, log: (String) -> Void) throws {
        log("-- sustained \(Int(seconds)) s, thermal bắt đầu: \(Self.thermal()) --")
        let start = Self.now(); var n = 0, nextLog = 15.0, aud = 0.0, cmp = 0.0
        while Self.now() - start < seconds, !shouldStop() {
            let r = try synth(res.texts[n % res.texts.count]); n += 1; aud += r.audio; cmp += r.total
            if Self.now() - start >= nextLog {
                log(String(format: "t=%3.0fs thermal %@ RTF %.2f footprint %d MB", Self.now() - start, Self.thermal(), cmp / aud, Int(Self.footprintMB())))
                nextLog += 15; aud = 0; cmp = 0
            }
        }
        log("kết thúc, thermal: \(Self.thermal())")
    }

    private func synth(_ text: String) throws -> Run { try autoreleasepool { try synthUnpooled(text) } }

    private func synthUnpooled(_ text: String) throws -> Run {
        var r = Run(); let start = Self.now()
        let ph = try g2p.phonemize(text, puncNorm: true); r.g2p = Self.now() - start
        let rows = embedder.buildRows(phonemeIDs: tokenizer.encode(ph), refCodes: refRows); r.promptRows = rows.count
        try codec.reset()
        var pending: [[Int]] = [], samples = 0, first = false, tPre = 0.0
        func flush() throws {
            guard !pending.isEmpty else { return }
            samples += try codec.decode(pending).count; pending = []
            if !first { first = true; r.ttfa = Self.now() - start }
        }
        let res = try generator.generate(rows: rows, anchor: anchor, frameCap: 300, onPrefillDone: { tPre = Self.now() }, onFrame: { _, codes, last in
            pending.append(codes)
            if pending.count >= (first ? Self.group : Self.firstGroup) || last { try flush() }
        })
        r.prefill = tPre - start - r.g2p; r.total = Self.now() - start; r.gen = r.total - r.g2p - r.prefill
        r.frames = res.frames.count; r.audio = Double(samples) / 48000
        return r
    }

    // MARK: helpers
    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
    static func ms(_ s: Double) -> String { String(format: "%.0f ms", s * 1000) }
    public static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"; case .fair: "fair"; case .serious: "serious"; case .critical: "critical"; @unknown default: "?"
        }
    }
    /// phys_footprint — con số iOS dùng để quyết định jetsam.
    public static func footprintMB() -> Double {
        var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let r = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
        return r == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }
}
