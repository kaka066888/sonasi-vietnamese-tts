import Foundation

/// Đo văn bản dài: TTFA (đến phần audio đầu tiên), RTF, RAM, và mô phỏng phát liên tục để đếm khựng (underrun).
public enum VieNeuLongBenchmark {
    public struct Report: Sendable {
        public var characters = 0, chunks = 0, retries = 0
        public var audioSeconds = 0.0, computeSeconds = 0.0
        public var ttfa = 0.0                       // giây từ lúc gọi tới phần audio đầu tiên
        public var stallSeconds = 0.0, stallCount = 0
        public var peakFootprintMB = 0.0, startThermal = "", endThermal = ""
        public var ttfaBreakdown: [(String, Double)] = []
        public var rtf: Double { audioSeconds > 0 ? computeSeconds / audioSeconds : 0 }
    }

    /// `onAudio` nhận từng phần audio (để phát); `log` nhận dòng kết quả.
    public static func run(_ synth: VieNeuSynthesizer, text: String, options: VieNeuSynthesizer.Options = .init(),
                           onAudio: (([Float]) -> Void)? = nil, shouldStop: @escaping () -> Bool = { false },
                           log: @escaping (String) -> Void) throws -> Report {
        var r = Report(); r.characters = text.count; r.startThermal = VieNeuBenchmark.thermal()
        let start = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }
        var arrivals: [(t: Double, dur: Double)] = []
        var chunkIndex = 0
        r.peakFootprintMB = VieNeuBenchmark.footprintMB()
        final class Events: @unchecked Sendable { var list: [(String, Double)] = [] }
        let events = Events()
        synth.trace = { events.list.append(($0, $1)) }
        defer { synth.trace = nil }
        let out = try synth.synthesize(text, options: options, onAudio: { piece in
            if shouldStop() { throw CancellationError() }
            guard !piece.isEmpty else { return }
            let t = elapsed()
            if arrivals.isEmpty { r.ttfa = t }
            arrivals.append((t, Double(piece.count) / Double(VieNeuAudio.sampleRate)))
            r.peakFootprintMB = max(r.peakFootprintMB, VieNeuBenchmark.footprintMB())
            onAudio?(piece)
        }, onChunk: { info in
            chunkIndex += 1
            log(String(format: "  chunk %2d: %3d frame (trần %3d) %5.2f s tính%@  «%@»", chunkIndex, info.frames, info.capFrames, info.seconds,
                       info.retries > 0 ? " · sinh lại \(info.retries)" : "", String(info.text.prefix(40))))
        })
        r.ttfaBreakdown = events.list
        r.computeSeconds = elapsed()
        r.audioSeconds = out.seconds; r.chunks = out.chunks.count; r.retries = out.chunks.reduce(0) { $0 + $1.retries }
        r.endThermal = VieNeuBenchmark.thermal()
        // Mô phỏng phát liên tục bắt đầu ngay tại TTFA: phần k cần có mặt trước (tổng thời lượng trước nó + TTFA + khựng đã tích lũy).
        var played = 0.0, stall = 0.0
        for a in arrivals {
            let needBy = r.ttfa + played + stall
            if a.t > needBy { stall += a.t - needBy; r.stallCount += 1 }
            played += a.dur
        }
        r.stallSeconds = stall
        return r
    }

    public static func format(_ r: Report) -> [String] {
        var prev = 0.0
        let breakdown = r.ttfaBreakdown.map { e -> String in defer { prev = e.1 }; return String(format: "    %6.0f ms (+%3.0f)  %@", e.1 * 1000, (e.1 - prev) * 1000, e.0) }
        return [String(format: "%d ký tự → %d chunk, audio %.1f s, tính %.1f s → RTF %.2f", r.characters, r.chunks, r.audioSeconds, r.computeSeconds, r.rtf),
         String(format: "TTFA (đến phần audio đầu) %.0f ms", r.ttfa * 1000),
         String(format: "phát liên tục từ TTFA: khựng %d lần, tổng %.2f s%@", r.stallCount, r.stallSeconds, r.stallCount == 0 ? " ✅" : ""),
         String(format: "footprint đỉnh %d MB · thermal %@ → %@ · sinh lại %d lần", Int(r.peakFootprintMB), r.startThermal, r.endThermal, r.retries)]
            + ["TTFA chia theo giai đoạn (chunk đầu):"] + breakdown
    }
}
