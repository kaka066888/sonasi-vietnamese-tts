import Foundation
import Observation
import SonasiVietnameseKit
import UIKit

final class StopFlag: @unchecked Sendable {
    private let lock = NSLock(); private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
}

@Observable @MainActor
final class BenchModel {
    enum Mode: String, CaseIterable, Identifiable { case sentences = "Một câu", longText = "Văn bản dài"; var id: String { rawValue } }
    struct LongText: Decodable, Hashable { let name: String; let text: String }

    var threads = 1
    var prefillThreads = 2      // prefill là tính toán dày nên chạy nhiều luồng hơn phần sinh frame
    var mode = Mode.longText
    var playAudio = true
    var textIndex = 1
    var longTexts: [LongText] = []
    var running = false
    var log = ""
    var thermal = VieNeuBenchmark.thermal()
    let deviceInfo: String

    private let stopFlag = StopFlag()
    private var observer: NSObjectProtocol?

    init() {
        var sys = utsname(); uname(&sys)
        let machine = withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) } }
        let gb = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        deviceInfo = String(format: "%@ · iOS %@ · RAM %.1f GB · %d lõi", machine, UIDevice.current.systemVersion, gb, ProcessInfo.processInfo.activeProcessorCount)
        observer = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.thermal = VieNeuBenchmark.thermal()
                self.append("[thermal đổi → \(self.thermal)]")
            }
        }
        if let u = Bundle.main.url(forResource: "long_texts", withExtension: "json"),
           let d = try? JSONDecoder().decode([LongText].self, from: Data(contentsOf: u)) { longTexts = d }
        append(deviceInfo + (ProcessInfo.processInfo.isLowPowerModeEnabled ? "\n⚠️ Low Power Mode đang bật — tắt trước khi đo." : ""))
    }

    func append(_ line: String) { log += (log.isEmpty ? "" : "\n") + line }

    func stop() { stopFlag.set(true) }

    func start(sustained: Double?) {
        guard !running else { return }
        running = true; stopFlag.set(false)
        UIApplication.shared.isIdleTimerDisabled = true
        append("\n=== \(sustained == nil ? "benchmark" : "sustained") · \(threads) luồng · int8 ===")
        let threads = threads, stop = stopFlag
        Task.detached(priority: .userInitiated) { [weak self] in
            let emit: @Sendable (String) -> Void = { line in Task { @MainActor in self?.append(line); self?.thermal = VieNeuBenchmark.thermal() } }
            do {
                let b = Bundle.main
                func url(_ n: String, _ e: String? = nil) throws -> URL {
                    guard let u = b.url(forResource: n, withExtension: e) else { throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "thiếu tài nguyên \(n)"]) }
                    return u
                }
                let texts = try JSONDecoder().decode([String].self, from: Data(contentsOf: url("bench_texts", "json")))
                let res = VieNeuBenchmark.Resources(modelDirectory: try url("onnx_int8"), codecDirectory: try url("codec"),
                                                    dictionary: try url("sea_g2p", "bin"), voice: try url("voice", "npz"), texts: texts)
                emit("đang nạp model…")
                let bench = try VieNeuBenchmark(res, threads: threads)
                if let s = sustained { try bench.sustained(seconds: s, shouldStop: { stop.isSet }, log: emit) }
                else { try bench.run(log: emit) }
            } catch {
                emit("LỖI: \(error)")
            }
            await MainActor.run { self?.running = false; UIApplication.shared.isIdleTimerDisabled = false }
        }
    }

    /// Văn bản dài qua VieNeuSynthesizer: TTFA, RTF, khựng khi phát liên tục, RAM; tuỳ chọn phát audio streaming thật.
    /// `repeatFor` > 0: lặp liên tục tới hết số giây (đo nhiệt khi tải thật).
    func startLong(repeatFor: Double?) {
        guard !running, longTexts.indices.contains(textIndex) else { return }
        running = true; stopFlag.set(false)
        UIApplication.shared.isIdleTimerDisabled = true
        let item = longTexts[textIndex]
        append("\n=== văn bản dài · \(item.name) · \(threads) luồng (prefill \(prefillThreads)) · int8\(playAudio ? " · phát audio" : "") ===")
        let threads = threads, prefillThreads = prefillThreads, stop = stopFlag, play = playAudio
        Task.detached(priority: .userInitiated) { [weak self] in
            let emit: @Sendable (String) -> Void = { line in Task { @MainActor in self?.append(line); self?.thermal = VieNeuBenchmark.thermal() } }
            let player = StreamPlayer()
            do {
                let b = Bundle.main
                func url(_ n: String, _ e: String? = nil) throws -> URL {
                    guard let u = b.url(forResource: n, withExtension: e) else { throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "thiếu tài nguyên \(n)"]) }
                    return u
                }
                emit("đang nạp model…")
                let t0 = Date()
                let synth = try VieNeuSynthesizer(modelDirectory: try url("onnx_int8"), codecDirectory: try url("codec"),
                                                  dictionary: try url("sea_g2p", "bin"), voice: try url("voice", "npz"), threads: threads, prefillThreads: prefillThreads)
                emit(String(format: "nạp xong %.1f s, footprint %d MB, thermal %@", Date().timeIntervalSince(t0), Int(VieNeuBenchmark.footprintMB()), VieNeuBenchmark.thermal()))
                _ = try synth.synthesize("Xin chào.")           // warm-up
                if play { try player.start() }
                let sink: ([Float]) -> Void = { play ? player.schedule($0) : () }
                if let total = repeatFor {
                    let start = Date(); var n = 0
                    while Date().timeIntervalSince(start) < total, !stop.isSet {
                        n += 1
                        let r = try VieNeuLongBenchmark.run(synth, text: item.text, onAudio: sink, shouldStop: { stop.isSet }, log: { _ in })
                        emit(String(format: "#%d t=%3.0fs RTF %.2f TTFA %.0f ms khựng %d (%.1f s) footprint %d MB thermal %@", n, Date().timeIntervalSince(start),
                                    r.rtf, r.ttfa * 1000, r.stallCount, r.stallSeconds, Int(r.peakFootprintMB), r.endThermal))
                    }
                    emit("kết thúc, thermal: \(VieNeuBenchmark.thermal())")
                } else {
                    let r = try VieNeuLongBenchmark.run(synth, text: item.text, onAudio: sink, shouldStop: { stop.isSet }, log: emit)
                    VieNeuLongBenchmark.format(r).forEach(emit)
                }
            } catch is CancellationError {
                emit("đã dừng")
            } catch {
                emit("LỖI: \(error)")
            }
            player.stop()
            await MainActor.run { self?.running = false; UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}
