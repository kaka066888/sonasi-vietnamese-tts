import SonasiVietnameseKit
import SwiftUI

struct ContentView: View {
    @State private var model = BenchModel()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text(model.deviceInfo).font(.caption).foregroundStyle(.secondary)
                Picker("Chế độ", selection: $model.mode) {
                    ForEach(BenchModel.Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).disabled(model.running)
                HStack {
                    Picker("Luồng", selection: $model.threads) {
                        ForEach([1, 2, 3, 4, 6], id: \.self) { Text("\($0) luồng").tag($0) }
                    }
                    .pickerStyle(.menu).disabled(model.running)
                    Spacer()
                    Text("Nhiệt: \(model.thermal)").font(.caption.monospaced())
                }
                if model.mode == .longText {
                    Picker("Prefill", selection: $model.prefillThreads) {
                        ForEach([1, 2, 3, 4], id: \.self) { Text("prefill \($0) luồng").tag($0) }
                    }
                    .pickerStyle(.menu).disabled(model.running)
                    Picker("Văn bản", selection: $model.textIndex) {
                        ForEach(Array(model.longTexts.enumerated()), id: \.offset) { Text($1.name).tag($0) }
                    }
                    .pickerStyle(.menu).disabled(model.running)
                    Toggle("Phát audio (streaming)", isOn: $model.playAudio).disabled(model.running)
                    HStack {
                        Button("Chạy") { model.startLong(repeatFor: nil) }
                        Button("Liên tục 2 phút") { model.startLong(repeatFor: 120) }
                        Button("Dừng", role: .destructive) { model.stop() }.disabled(!model.running)
                    }
                    .buttonStyle(.borderedProminent).disabled(model.running && false)
                } else {
                    HStack {
                        Button("Chạy benchmark") { model.start(sustained: nil) }
                        Button("Sustained 2 phút") { model.start(sustained: 120) }
                        Button("Dừng", role: .destructive) { model.stop() }.disabled(!model.running)
                    }
                    .buttonStyle(.borderedProminent).disabled(model.running && false)
                }
                if model.running { ProgressView().frame(maxWidth: .infinity) }
                ScrollView {
                    Text(model.log).font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .background(Color(.secondarySystemBackground)).clipShape(RoundedRectangle(cornerRadius: 8))
            }
            .padding()
            .navigationTitle("Sonasi TTS Bench")
            .toolbar { ToolbarItem(placement: .primaryAction) { ShareLink(item: model.log) } }
        }
    }
}
