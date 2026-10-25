import Foundation
import OnnxRuntimeBindings

/// Vòng sinh frame: prefill → (acoustic head 16 kênh + backbone decode_step) lặp lại, greedy.
public final class VieNeuGenerator: @unchecked Sendable {
    public struct Result {
        public let frames: [[Int]]      // gồm cả frame cuối khi gặp eos (giống bản Python)
        public let eosAt: Int?          // chỉ số frame có eos, nil nếu chạm cap
    }

    public static let defaultRepWindow = 64

    /// Lấy mẫu như engine Python (mặc định temperature 0,8, top-k 25, top-p 0,95). `nil` = greedy.
    public struct Sampling: Sendable {
        public var temperature: Float, topK: Int, topP: Float, seed: UInt64?
        public init(temperature: Float = 0.8, topK: Int = 25, topP: Float = 0.95, seed: UInt64? = nil) {
            self.temperature = temperature; self.topK = topK; self.topP = topP; self.seed = seed
        }
    }

    struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private let embedder: VieNeuEmbedder
    private let prefill: VieNeuPrefill
    private let decode: ORTSession
    private let acoustic: ORTSession
    private let decodeOutNames: [String]
    private let acousticOutNames: [String]
    private let layers = 12, localHeads = 8
    private let eosToken: Int, speechStart: Int

    /// `prefillThreads`: prefill là tính toán dày (hưởng lợi từ nhiều luồng), còn decode/acoustic từng frame thì không — nil = như `threads`.
    public init(embedder: VieNeuEmbedder, modelDirectory dir: URL, threads: Int = 1, prefillThreads: Int? = nil) throws {
        self.embedder = embedder
        prefill = try VieNeuPrefill(modelURL: dir.appendingPathComponent("vieneu_prefill.onnx"), threads: prefillThreads ?? threads)
        decode = try ORT.session(dir.appendingPathComponent("vieneu_decode_step.onnx"), threads: threads)
        // Graph acoustic chứa MatMulInteger/DynamicQuantizeLinear (cả bản "fp32"): kết quả phụ thuộc cách ORT fuse
        // và kernel của nền tảng. Trên macOS chỉ `.none` cho kết quả gần Python/Linux; xem ORT.session.
        acoustic = try ORT.session(dir.appendingPathComponent("vieneu_acoustic_cached.onnx"), threads: threads, level: .none)
        decodeOutNames = try decode.outputNames()
        acousticOutNames = try acoustic.outputNames()
        let c = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
        eosToken = c["speech_generation_end_token_id"] as? Int ?? 6
        speechStart = c["speech_generation_start_token_id"] as? Int ?? 5
    }

    /// Greedy (temperature 0): argmax sau repetition penalty (cửa sổ trượt `repWindow` frame mỗi kênh).
    /// `onPrefillDone` gọi ngay sau prefill; `onFrame` gọi sau mỗi frame (kể cả frame eos) để phát audio streaming.
    public func generate(rows: [[Int]], anchor: [Float]?, frameCap: Int,
                         repPenalty: Float = 1.2, repWindow: Int = defaultRepWindow, sampling: Sampling? = nil,
                         onPrefillDone: (() -> Void)? = nil,
                         onFrame: ((_ index: Int, _ codes: [Int], _ isLast: Bool) throws -> Void)? = nil) throws -> Result {
        let nVQ = embedder.config.nVQ, hdim = embedder.config.hidden
        let t0 = rows.count
        let pre = try prefill.run(embeds: embedder.embed(rows: rows, anchor: anchor), length: t0)
        onPrefillDone?()
        var pastK = pre.pastK, pastV = pre.pastV
        var h = pre.lastHidden
        var hist = [[Int]](repeating: [], count: nVQ)   // FIFO theo thứ tự sinh
        var frames: [[Int]] = []
        var rng = SplitMix64(state: sampling?.seed ?? UInt64.random(in: .min ... .max))

        for t in 0..<frameCap {
            let (codes, eos) = try acousticFrame(h, hist: &hist, repPenalty: repPenalty, window: repWindow, sampling: sampling, rng: &rng)
            frames.append(codes)
            try onFrame?(t, codes, eos)
            if eos { return Result(frames: frames, eosAt: t) }

            let slot = [speechStart] + codes
            var feed: [String: ORTValue] = [
                "inputs_embeds": try ORT.tensor(embedder.embed(rows: [slot], anchor: anchor), [1, 1, hdim]),
                "position_ids": try ORT.tensor([Int64(t0 + t)], [1, 1]),
            ]
            for j in 0..<layers { feed["past_k_\(j)"] = pastK[j]; feed["past_v_\(j)"] = pastV[j] }
            let out = try ORT.run(decode, feed, decodeOutNames)
            h = try ORT.floats(out[decodeOutNames[0]]!)
            pastK = (0..<layers).map { out[decodeOutNames[1 + $0]]! }
            pastV = (0..<layers).map { out[decodeOutNames[1 + layers + $0]]! }
        }
        return Result(frames: frames, eosAt: nil)
    }

    private func acousticFrame(_ cond: [Float], hist: inout [[Int]], repPenalty: Float, window: Int,
                               sampling: Sampling?, rng: inout SplitMix64) throws -> ([Int], Bool) {
        let hdim = embedder.config.hidden, nVQ = embedder.config.nVQ
        let empty = try ORT.tensor([Float](), [1, localHeads, 0, hdim / localHeads])
        var feed: [String: ORTValue] = [
            "token_emb": try ORT.tensor(cond + embedder.textRow(speechStart), [1, 2, hdim]),
            "position_ids": try ORT.tensor([Int64(0), 1], [1, 2]),
            "past_k_0": empty, "past_v_0": empty,
        ]
        var out = try ORT.run(acoustic, feed, acousticOutNames)
        var hidden = try ORT.floats(out[acousticOutNames[0]]!)       // (1, 2, H)
        let slot0 = Array(hidden[0..<hdim])
        var codes: [Int] = []

        var localRNG = rng
        defer { rng = localRNG }
        func pick(_ ch: Int, _ vec: [Float]) -> Int {
            var logits = embedder.audioLogits(channel: ch, hidden: vec)
            for idx in Set(hist[ch]) {
                let s = logits[idx]
                logits[idx] = s < 0 ? s * repPenalty : s / repPenalty
            }
            let best: Int
            if let s = sampling, s.temperature > 0 { best = Self.sample(logits, s, &localRNG) }
            else { var b = 0; for i in 1..<logits.count where logits[i] > logits[b] { b = i }; best = b }
            hist[ch].append(best)
            if window > 0, hist[ch].count > window { hist[ch].removeFirst() }
            return best
        }

        codes.append(pick(0, Array(hidden[hdim..<(2 * hdim)])))
        for ch in 1..<nVQ {
            feed = [
                "token_emb": try ORT.tensor(embedder.audioRow(channel: ch - 1, code: codes[ch - 1]), [1, 1, hdim]),
                "position_ids": try ORT.tensor([Int64(ch + 1)], [1, 1]),
                "past_k_0": out[acousticOutNames[1]]!, "past_v_0": out[acousticOutNames[2]]!,
            ]
            out = try ORT.run(acoustic, feed, acousticOutNames)
            hidden = try ORT.floats(out[acousticOutNames[0]]!)
            codes.append(pick(ch, Array(hidden[0..<hdim])))
        }
        let tl = embedder.textLogits(hidden: slot0)
        var best = 0
        for i in 1..<tl.count where tl[i] > tl[best] { best = i }
        return (codes, best == eosToken)
    }

    /// temperature → top-k (sắp xếp giảm dần) → softmax → top-p (nucleus) → rút mẫu.
    static func sample(_ logitsIn: [Float], _ s: Sampling, _ rng: inout SplitMix64) -> Int {
        let v = logitsIn.count
        var idx = Array(0..<v)
        idx.sort { logitsIn[$0] > logitsIn[$1] }
        if s.topK > 0 && s.topK < v { idx = Array(idx.prefix(s.topK)) }
        let scaled = idx.map { logitsIn[$0] / s.temperature }
        let m = scaled[0]
        var p = scaled.map { expf($0 - m) }
        let z = p.reduce(0, +)
        p = p.map { $0 / z }
        if s.topP > 0 && s.topP < 1 {
            var cum: Float = 0
            for i in p.indices { let before = cum; cum += p[i]; if before >= s.topP { p[i] = 0 } }
            let z2 = p.reduce(0, +)
            p = p.map { $0 / z2 }
        }
        let u = Float.random(in: 0..<1, using: &rng)
        var acc: Float = 0
        for i in p.indices { acc += p[i]; if u < acc { return idx[i] } }
        return idx[p.lastIndex { $0 > 0 } ?? 0]
    }
}
