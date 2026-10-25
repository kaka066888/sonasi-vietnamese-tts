import Accelerate
import Foundation

/// Phần NumPy của engine Python: bảng embedding tied, speaker anchor, dựng rows và embed.
public final class VieNeuEmbedder: @unchecked Sendable {
    public struct Config: Sendable {
        public var nVQ = 16, hidden = 768
        public var audioPad = 1024, textPromptStart = 3, textPromptEnd = 4, refSlot = 7
        public var defaultStyle = 16
        public init(configJSON: Data) throws {
            guard let c = try JSONSerialization.jsonObject(with: configJSON) as? [String: Any] else { throw NPZ.Failure.malformed("config") }
            func i(_ k: String, _ d: Int) -> Int { c[k] as? Int ?? d }
            nVQ = i("n_vq", 16); hidden = i("hidden_size", 768)
            audioPad = i("audio_pad_token_id", 1024); textPromptStart = i("text_prompt_start_token_id", 3)
            textPromptEnd = i("text_prompt_end_token_id", 4); refSlot = i("audio_ref_slot_token_id", 7)
            defaultStyle = i("default_style_token_id", 16)
        }
    }

    public let config: Config
    private let textEmb: [Float]          // (Vt, H)
    private let audioEmb: [Float]         // (n_vq, Va, H)
    private let audioVocab: Int
    private let xvecW: [Float], xvecB: [Float], xvecLNW: [Float], xvecLNB: [Float]
    private let xvecLNEps: Float
    private let speakerDim: Int

    public init(config: Config, headsURL: URL) throws {
        self.config = config
        let z = try NPZ.read(headsURL)
        func get(_ k: String) throws -> NPYArray {
            guard let a = z[k] else { throw NPZ.Failure.malformed("heads.npz thiếu \(k)") }
            return a
        }
        textEmb = try get("text_emb").floats()
        let ae = try get("audio_emb")
        audioEmb = try ae.floats()
        audioVocab = ae.shape[1]
        let w = try get("xvec_w")
        xvecW = try w.floats(); speakerDim = w.shape[1]
        xvecB = try get("xvec_b").floats()
        xvecLNW = try get("xvec_ln_w").floats()
        xvecLNB = try get("xvec_ln_b").floats()
        xvecLNEps = try get("xvec_ln_eps").floats()[0]
    }

    /// 192-d x-vector → (H,): Linear + LayerNorm.
    public func speakerAnchor(_ emb: [Float]) -> [Float] {
        let h = config.hidden
        var v = xvecB
        for r in 0..<h {
            var acc: Float = 0
            xvecW.withUnsafeBufferPointer { w in
                emb.withUnsafeBufferPointer { e in vDSP_dotpr(w.baseAddress! + r * speakerDim, 1, e.baseAddress!, 1, &acc, vDSP_Length(speakerDim)) }
            }
            v[r] += acc
        }
        let mean = v.reduce(0, +) / Float(h)
        let variance = v.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(h)
        let inv = 1 / (variance + xvecLNEps).squareRoot()
        return (0..<h).map { (v[$0] - mean) * inv * xvecLNW[$0] + xvecLNB[$0] }
    }

    /// rows: T hàng × (n_vq+1) cột. Hàng văn bản + (tùy chọn) hàng ref codes.
    public func buildRows(phonemeIDs: [Int], refCodes: [[Int]]?) -> [[Int]] {
        let c = config
        let textIDs = [c.defaultStyle, c.textPromptStart] + phonemeIDs + [c.textPromptEnd]
        var rows = textIDs.map { [$0] + [Int](repeating: c.audioPad, count: c.nVQ) }
        for rc in refCodes ?? [] { rows.append([c.refSlot] + rc) }
        return rows
    }

    /// (T, H) phẳng, row-major.
    public func embed(rows: [[Int]], anchor: [Float]?) -> [Float] {
        let h = config.hidden, n = config.nVQ, pad = config.audioPad
        var out = [Float](repeating: 0, count: rows.count * h)
        out.withUnsafeMutableBufferPointer { o in
            for (t, row) in rows.enumerated() {
                let dst = o.baseAddress! + t * h
                textEmb.withUnsafeBufferPointer { te in
                    dst.update(from: te.baseAddress! + row[0] * h, count: h)
                }
                for ch in 0..<n {
                    let id = row[ch + 1]
                    if id == pad { continue }
                    audioEmb.withUnsafeBufferPointer { ae in
                        vDSP_vadd(dst, 1, ae.baseAddress! + (ch * audioVocab + id) * h, 1, dst, 1, vDSP_Length(h))
                    }
                }
                if let a = anchor { a.withUnsafeBufferPointer { vDSP_vadd(dst, 1, $0.baseAddress!, 1, dst, 1, vDSP_Length(h)) } }
            }
        }
        return out
    }

    public var audioVocabSize: Int { audioVocab }
    public var textVocabSize: Int { textEmb.count / config.hidden }

    public func textRow(_ id: Int) -> [Float] {
        let h = config.hidden
        return Array(textEmb[(id * h)..<((id + 1) * h)])
    }

    public func audioRow(channel: Int, code: Int) -> [Float] {
        let h = config.hidden
        let s = (channel * audioVocab + code) * h
        return Array(audioEmb[s..<(s + h)])
    }

    /// logits = vec · audio_emb[ch]ᵀ  → (audioVocab,)
    public func audioLogits(channel: Int, hidden vec: [Float]) -> [Float] {
        let h = config.hidden
        var out = [Float](repeating: 0, count: audioVocab)
        audioEmb.withUnsafeBufferPointer { a in
            vec.withUnsafeBufferPointer { v in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(audioVocab), Int32(h), 1,
                            a.baseAddress! + channel * audioVocab * h, Int32(h), v.baseAddress!, 1, 0, &out, 1)
            }
        }
        return out
    }

    /// logits văn bản = vec · text_embᵀ  → (textVocab,)
    public func textLogits(hidden vec: [Float]) -> [Float] {
        let h = config.hidden, n = textVocabSize
        var out = [Float](repeating: 0, count: n)
        textEmb.withUnsafeBufferPointer { a in
            vec.withUnsafeBufferPointer { v in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(n), Int32(h), 1, a.baseAddress!, Int32(h), v.baseAddress!, 1, 0, &out, 1)
            }
        }
        return out
    }
}
