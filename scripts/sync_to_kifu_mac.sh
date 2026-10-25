#!/usr/bin/env bash
# Chép SonasiVietnameseKit (chỉ phần thư viện) sang repo kifu_mac để app dùng như package cục bộ.
#   scripts/sync_to_kifu_mac.sh <đường-dẫn-kifu_mac> [--dry-run] [--no-framework]
# Nguồn sự thật là repo này (test, golden, bench, model nằm ở đây); bản trong kifu_mac chỉ là bản vendored,
# KHÔNG sửa trực tiếp ở đó — sửa ở đây rồi chạy lại script.
set -euo pipefail
DEST_ROOT=${1:?đường dẫn repo kifu_mac}
shift || true
DRY=0; FW=1
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --no-framework) FW=0;; *) echo "tham số lạ: $a"; exit 2;; esac; done
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC="$HERE/Packages/SonasiVietnameseKit"
DEST="$DEST_ROOT/Packages/SonasiVietnameseKit"
[ -d "$DEST_ROOT/Sonasi.xcodeproj" ] || { echo "$DEST_ROOT không giống repo kifu_mac (thiếu Sonasi.xcodeproj)"; exit 1; }
[ -d "$SRC/Frameworks/CSeaG2P.xcframework" ] || [ $FW -eq 0 ] || { echo "thiếu CSeaG2P.xcframework — chạy scripts/build_sea_g2p_xcframework.sh trước"; exit 1; }
RS=(rsync -a --delete)
[ $DRY -eq 1 ] && RS+=(-n -v)

mkdir -p "$DEST/Sources"
"${RS[@]}" "$SRC/Sources/SonasiVietnameseKit/" "$DEST/Sources/SonasiVietnameseKit/"
if [ $FW -eq 1 ]; then
  mkdir -p "$DEST/Frameworks"
  "${RS[@]}" "$SRC/Frameworks/CSeaG2P.xcframework/" "$DEST/Frameworks/CSeaG2P.xcframework/"
fi
if [ $DRY -eq 0 ]; then
cat > "$DEST/Package.swift" <<'EOF'
// swift-tools-version: 6.0
import PackageDescription

// VENDORED từ repo sonasi-vietnamese-tts (scripts/sync_to_kifu_mac.sh) — đừng sửa tay ở đây.
// TTS tiếng Việt chạy hoàn toàn trên máy: VieNeu-TTS v3 Turbo (ONNX Runtime) + sea-g2p (Rust, xcframework).
let package = Package(
    name: "SonasiVietnameseKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "SonasiVietnameseKit", targets: ["SonasiVietnameseKit"])],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        .binaryTarget(name: "CSeaG2P", path: "Frameworks/CSeaG2P.xcframework"),
        .target(name: "SonasiVietnameseKit", dependencies: [
            "CSeaG2P",
            .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
        ], swiftSettings: [
            // Debug của app host không tối ưu Swift (RTF gấp đôi); thư viện này là vòng lặp số học nên luôn -O.
            .unsafeFlags(["-O"], .when(configuration: .debug)),
        ]),
    ]
)
EOF
printf '.build/\n.swiftpm/\n' > "$DEST/.gitignore"
echo "OK → $DEST"
fi
