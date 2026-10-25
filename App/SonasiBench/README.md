# SonasiBench — đo TTS trên iPhone

App SwiftUI tối giản chạy `VieNeuBenchmark` (cùng code với CLI `SonasiBench` trên Mac): đo **TTFA**, **RTF**,
**thời gian mỗi frame**, **RAM (phys_footprint)** và **thermalState** của VieNeu-TTS v3 Turbo **int8**.

## Chuẩn bị (một lần)

1. Mac có Xcode, iPhone chạy iOS 17 trở lên, cáp USB.
2. iPhone: **Settings → Privacy & Security → Developer Mode → bật** (máy sẽ khởi động lại).
3. Xcode → **Settings → Accounts** → thêm Apple ID (tài khoản miễn phí đủ để cài lên máy của chính bạn).
4. Kiểm tra các file model có trong repo (app đóng gói thẳng từ đây, ~300 MB):
   - `Models/onnx_int8/` (config, tokenizer, 3 file `.onnx`, `vieneu_backbone_shared.data`, `vieneu_v3_heads.npz`)
   - `Models/codec/` (`moss_audio_tokenizer_decode_step.onnx`, `*.data`, `codec_browser_onnx_meta.json`)
   - `App/SonasiBench/Resources/` (`sea_g2p.bin`, `voice.npz`, `bench_texts.json`) — đã có sẵn.
   - `Packages/SonasiVietnameseKit/Frameworks/CSeaG2P.xcframework` (build bằng `scripts/build_sea_g2p_xcframework.sh <repo sea-g2p>`; macOS universal + iOS).

## Build lên iPhone

```bash
cd App/SonasiBench
xcodegen generate        # chỉ cần khi sửa project.yml; file .xcodeproj đã được sinh sẵn
open SonasiBench.xcodeproj
```

Trong Xcode:

1. Chọn project **SonasiBench** → target **SonasiBench** → **Signing & Capabilities**.
   - Tick **Automatically manage signing**, chọn **Team** của bạn.
   - Nếu báo trùng Bundle Identifier, đổi `app.sonasi.bench` thành chuỗi riêng (ví dụ `com.tenban.sonasibench`).
2. Cắm iPhone, chọn nó ở thanh thiết bị phía trên (không chọn Simulator). Lần đầu bấm **Trust** trên iPhone.
3. **Product → Run** (⌘R). Scheme đã đặt sẵn **Release** — đừng đổi sang Debug, Debug làm phần Swift chậm giả tạo.
4. Lần đầu iPhone báo "Untrusted Developer": **Settings → General → VPN & Device Management** → chọn Apple ID → **Trust**, rồi mở lại app.

Build lần đầu mất vài phút (ORT + cài ~300 MB lên máy).

## Cách đo cho kết quả đáng tin

- Tắt **Low Power Mode** (app có cảnh báo nếu đang bật). Tắt Chế độ Tối ưu hóa pin sạc nếu có.
- Rút cáp khi đo **sustained** (sạc làm máy nóng thêm, nhưng cắm cáp thì Xcode mất log; số liệu đã nằm ngay trong app).
- Để máy nguội trước mỗi lượt (thermal `nominal`). Đóng các app nặng khác.
- Chạy theo thứ tự: **2 luồng** → **1 luồng** → **4 luồng** (nếu máy có ≥ 4 lõi hiệu năng), mỗi cấu hình một lượt **Chạy benchmark**
  (lượt đầu sau khi mở app có thể chậm hơn do nạp model, app đã warm-up 1 câu).
- Sau đó **Sustained 2 phút** với cấu hình tốt nhất, theo dõi dòng `thermal` và `RTF` mỗi 15 giây.
- Nút chia sẻ (góc trên phải) gửi toàn bộ log ra Notes/AirDrop/Messages.

## Đọc kết quả

| Chỉ số | Ý nghĩa | Mục tiêu gợi ý |
|---|---|---|
| **RTF** | thời gian tính / thời lượng audio; < 1 là nhanh hơn thời gian thực | ≤ 0,5 (có dư địa cho ASR chạy song song) |
| **TTFA** | từ lúc có text đến khi có 2 frame đầu (≈ 160 ms audio) đã giải mã xong | ≤ 300 ms |
| **mỗi frame** | tính cho 1 frame = 80 ms audio | < 80 ms (đủ phát liên tục), tốt hơn < 40 ms |
| **footprint** | RAM mà iOS dùng để quyết định kill app (jetsam) | iPhone 6 GB ≈ giới hạn thực tế ~3 GB; 4 GB còn thấp hơn |
| **thermal** | `nominal` → `fair` → `serious` → `critical` | giữ `nominal`/`fair`; RTF tăng khi `serious` là dấu hiệu bị throttle |

Mốc tham chiếu trên Mac (M4 Pro, 2 luồng, int8): RTF 0,13, TTFA ≈ 62 ms, ≈ 9 ms/frame, đỉnh 1,35 GB.
Simulator iOS trên cùng Mac: RTF 0,16, TTFA ≈ 76 ms, đỉnh 1,33 GB — **chạy bằng CPU Mac, không đại diện cho iPhone**.

## Nếu app bị kill giữa chừng (màn hình trắng/thoát)

Đó thường là **jetsam do hết RAM** (đỉnh khoảng 1,3 GB khi chạy). Cách xử lý:
- Xem log: Xcode → **Window → Devices and Simulators** → chọn iPhone → **Open Console**, tìm `jetsam`.
- Với tài khoản Apple Developer trả phí có thể thêm entitlement `com.apple.developer.kernel.increased-memory-limit`
  (Signing & Capabilities → **+ Capability** → *Increased Memory Limit*) để nâng trần RAM.
- Báo lại số `footprint` cuối cùng đọc được trong log để mình tính phương án giảm RAM (ví dụ nạp `.data` bằng mmap, bỏ bản sao embedding).

## Cấu trúc

- `project.yml` — spec XcodeGen (sửa rồi chạy `xcodegen generate`).
- `Sources/` — `SonasiBenchApp.swift`, `ContentView.swift` (UI), `BenchModel.swift` (chạy nền, thermal observer).
- Toàn bộ logic đo nằm ở `Packages/SonasiVietnameseKit/Sources/SonasiVietnameseKit/VieNeuBenchmark.swift`.
