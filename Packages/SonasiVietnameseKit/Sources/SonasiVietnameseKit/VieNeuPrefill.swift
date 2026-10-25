import Foundation
import OnnxRuntimeBindings

/// Phiên ORT cho graph prefill (CPU EP). Tạo từ đường dẫn file để ORT tìm được `.data` bên cạnh.
public final class VieNeuPrefill: @unchecked Sendable {
    public struct Output {
        public let lastHidden: [Float]      // (H,)
        public let pastK: [ORTValue]
        public let pastV: [ORTValue]
    }

    private let session: ORTSession
    private let layers: Int
    private let hidden: Int

    public init(modelURL: URL, layers: Int = 12, hidden: Int = 768, threads: Int = 1) throws {
        session = try ORT.session(modelURL, threads: threads)
        self.layers = layers; self.hidden = hidden
    }

    public func run(embeds: [Float], length t: Int) throws -> Output {
        let input = try ORT.tensor(embeds, [1, t, hidden])
        let outNames = try session.outputNames()
        let out = try ORT.run(session, ["inputs_embeds": input], outNames)
        guard let h = out[outNames[0]] else { throw NPZ.Failure.malformed("prefill output") }
        let all = try ORT.floats(h)
        let last = Array(all[((t - 1) * hidden)..<(t * hidden)])
        return Output(lastHidden: last,
                      pastK: (0..<layers).compactMap { out[outNames[1 + $0]] },
                      pastV: (0..<layers).compactMap { out[outNames[1 + layers + $0]] })
    }

    public func outputNames() throws -> [String] { try session.outputNames() }
    public func inputNames() throws -> [String] { try session.inputNames() }
}
