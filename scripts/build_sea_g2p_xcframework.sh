#!/usr/bin/env bash
# Build sea-g2p (C ABI) thành CSeaG2P.xcframework cho macOS (universal arm64+x86_64) + iOS device + iOS simulator.
# Chạy trên Mac có Xcode + rustup.
#   scripts/build_sea_g2p_xcframework.sh <đường-dẫn-repo-sea-g2p> [thư-mục-output]
set -euo pipefail
SRC=${1:?đường dẫn tới repo sea-g2p}
OUT=${2:-Packages/SonasiVietnameseKit/Frameworks}
TARGETS=(aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios)
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}
export IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET:-17.0}

rustup target add "${TARGETS[@]}"
for T in "${TARGETS[@]}"; do
  (cd "$SRC" && cargo rustc --release --target "$T" --no-default-features --features capi --crate-type staticlib)
done

L="$SRC/target"
TMP=$(mktemp -d)
lipo -create \
  "$L/aarch64-apple-darwin/release/libsea_g2p_rs.a" \
  "$L/x86_64-apple-darwin/release/libsea_g2p_rs.a" \
  -output "$TMP/libsea_g2p_rs_macos.a"
lipo -create \
  "$L/aarch64-apple-ios-sim/release/libsea_g2p_rs.a" \
  "$L/x86_64-apple-ios/release/libsea_g2p_rs.a" \
  -output "$TMP/libsea_g2p_rs.a"
mkdir -p "$TMP/include"
cp "$SRC/include/sea_g2p.h" "$TMP/include/"
printf 'module CSeaG2P {\n  header "sea_g2p.h"\n  export *\n}\n' > "$TMP/include/module.modulemap"

mkdir -p "$OUT"
rm -rf "$OUT/CSeaG2P.xcframework"
xcodebuild -create-xcframework \
  -library "$TMP/libsea_g2p_rs_macos.a"                      -headers "$TMP/include" \
  -library "$L/aarch64-apple-ios/release/libsea_g2p_rs.a"    -headers "$TMP/include" \
  -library "$TMP/libsea_g2p_rs.a"                            -headers "$TMP/include" \
  -output "$OUT/CSeaG2P.xcframework"
echo "OK → $OUT/CSeaG2P.xcframework"
echo "Nhớ kèm sea_g2p.bin (từ điển, $SRC/python/sea_g2p/sea_g2p.bin) — tải về lúc chạy, không nhúng vào app."
