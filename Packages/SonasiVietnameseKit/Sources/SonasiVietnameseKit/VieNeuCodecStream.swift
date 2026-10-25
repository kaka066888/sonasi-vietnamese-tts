import Foundation
import OnnxRuntimeBindings

/// MOSS audio tokenizer `decode_step`: giải mã từng nhóm frame, giữ state (ring-buffer cache) giữa các lần gọi.
/// Đầu ra: mono 48 kHz (trung bình 2 kênh).
public final class VieNeuCodecStream: @unchecked Sendable {
    private struct Spec: Decodable {
        struct Offset: Decodable { let input_name: String; let output_name: String; let shape: [Int] }
        struct Attn: Decodable {
            let offset_input_name: String, offset_output_name: String
            let cached_keys_input_name: String, cached_keys_output_name: String
            let cached_values_input_name: String, cached_values_output_name: String
            let cached_positions_input_name: String, cached_positions_output_name: String
            let offset_shape: [Int], cache_shape: [Int], positions_shape: [Int]
        }
        let transformer_offsets: [Offset]
        let attention_caches: [Attn]
    }

    private let session: ORTSession
    private let outNames: [String]
    private let spec: Spec
    private var state: [String: ORTValue] = [:]

    /// `directory` chứa moss_audio_tokenizer_decode_step.onnx (+ .data) và codec_browser_onnx_meta.json.
    public init(directory: URL, threads: Int = 1) throws {
        session = try ORT.session(directory.appendingPathComponent("moss_audio_tokenizer_decode_step.onnx"), threads: threads)
        outNames = try session.outputNames()
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("codec_browser_onnx_meta.json"))) as? [String: Any]
        guard let sd = meta?["streaming_decode"] else { throw NPZ.Failure.malformed("streaming_decode") }
        spec = try JSONDecoder().decode(Spec.self, from: JSONSerialization.data(withJSONObject: sd))
        try reset()
    }

    public func reset() throws {
        state = [:]
        func n(_ s: [Int]) -> Int { s.reduce(1, *) }
        for t in spec.transformer_offsets { state[t.input_name] = try ORT.tensor([Int32](repeating: 0, count: n(t.shape)), t.shape) }
        for a in spec.attention_caches {
            state[a.offset_input_name] = try ORT.tensor([Int32](repeating: 0, count: n(a.offset_shape)), a.offset_shape)
            state[a.cached_keys_input_name] = try ORT.tensor([Float](repeating: 0, count: n(a.cache_shape)), a.cache_shape)
            state[a.cached_values_input_name] = try ORT.tensor([Float](repeating: 0, count: n(a.cache_shape)), a.cache_shape)
            state[a.cached_positions_input_name] = try ORT.tensor([Int32](repeating: -1, count: n(a.positions_shape)), a.positions_shape)
        }
    }

    public func decode(_ frames: [[Int]]) throws -> [Float] {
        guard let nVQ = frames.first?.count else { return [] }
        var feed = state
        feed["audio_codes"] = try ORT.tensor(frames.flatMap { $0.map(Int32.init) }, [1, frames.count, nVQ])
        feed["audio_code_lengths"] = try ORT.tensor([Int32(frames.count)], [1])
        let out = try ORT.run(session, feed, outNames)
        for t in spec.transformer_offsets { state[t.input_name] = out[t.output_name]! }
        for a in spec.attention_caches {
            state[a.offset_input_name] = out[a.offset_output_name]!
            state[a.cached_keys_input_name] = out[a.cached_keys_output_name]!
            state[a.cached_values_input_name] = out[a.cached_values_output_name]!
            state[a.cached_positions_input_name] = out[a.cached_positions_output_name]!
        }
        let audio = try ORT.floats(out["audio"]!)
        let shape = try ORT.shape(out["audio"]!)                       // (1, C, N)
        let valid = Int(try ORT.int32s(out["audio_lengths"]!)[0])
        let ch = shape[1], n = shape[2]
        return (0..<valid).map { i in
            var s: Float = 0
            for c in 0..<ch { s += audio[c * n + i] }
            return s / Float(ch)
        }
    }
}
