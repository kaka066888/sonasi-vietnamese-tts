# SonasiVietnameseKit

On-device Vietnamese text-to-speech for Apple platforms, written in Swift.

SonasiVietnameseKit runs the [VieNeu-TTS v3 Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo)
model (int8) through [ONNX Runtime](https://onnxruntime.ai) on macOS and iOS. Synthesis happens entirely on the
device: no network, no server. Output is 48 kHz mono audio that can be streamed to a player while it is still
being generated.

> **Status:** experimental. The library, its parity tests and a benchmark app work on macOS and on an iPhone 15 Pro.
> There is no model downloader yet and only one voice is loaded at a time.

## Features

- Complete pipeline in Swift: text normalisation and phonemisation ([sea-g2p](https://github.com/pnnbao97/sea-g2p)),
  tokenizer, prefill, frame-by-frame generation (acoustic head + backbone), and the MOSS audio codec decoder.
- Long-text support: sentence splitting and chunk packing (up to 256 characters), pauses between chunks,
  a per-chunk frame cap, babble detection with automatic retry, and emotion tags
  (`[cười]`, `[thở dài]`, `[hắng giọng]`).
- Streaming output with low time-to-first-audio; the first chunk is shortened on purpose to start playback sooner.
- Verified against the reference Python implementation with golden vectors for the tokenizer, g2p, embeddings,
  prefill, decoding, codec and chunking.

## Performance

Measured with the int8 model in Release builds.

| Device | Workload | TTFA | RTF | Peak memory |
|---|---|---|---|---|
| Mac, Apple M4 Pro (1–2 threads) | single sentence | 72–127 ms | 0.13–0.15 | ~850 MB |
| iPhone 15 Pro (1 thread) | single sentence | 106 ms | 0.21 | ~580 MB |
| iPhone 15 Pro (1 thread) | 935-character passage, with playback | 126 ms | 0.25–0.26 | ~650 MB |
| iPhone 15 Pro (1 thread) | 133 s of continuous synthesis | 126–407 ms | 0.26 → 0.36 | 540 → 611 MB (flat) |

- **TTFA** is the time from the call to the first audio samples. **RTF** (real-time factor) is compute time divided
  by audio duration; below 1 is faster than real time.
- During 133 s of continuous synthesis on the iPhone the thermal state went from `nominal` to `fair` and never
  reached `serious`, with no audible gaps. With two generation threads the device reached `serious` after about
  80 s, so one generation thread is the recommended default for long reading sessions.
- Not measured yet: iPhones with 4–6 GB of RAM, concurrent load from other workloads (for example speech
  recognition), and battery drain.

## Requirements

| | |
|---|---|
| Platforms | macOS 14+, iOS 17+ |
| Tooling | Xcode with Swift 6, [Rust](https://rustup.rs) (to build sea-g2p), [XcodeGen](https://github.com/yonaskolb/XcodeGen) (benchmark app only) |
| Dependency | [onnxruntime-swift-package-manager](https://github.com/microsoft/onnxruntime-swift-package-manager) 1.24.2 (resolved by SwiftPM) |
| Disk | ~300 MB for the int8 model and codec, ~140 MB for the built xcframework |

## Repository layout

```
Packages/SonasiVietnameseKit/     Swift package: library, SonasiBench CLI, tests
  Sources/SonasiVietnameseKit/    tokenizer, g2p, prefill, generator, codec, chunker, synthesizer
  Sources/SonasiBench/            command-line benchmark for macOS
  Tests/                          parity tests and Tests/Golden reference data
  Frameworks/                     CSeaG2P.xcframework (you build it; not in git)
App/SonasiBench/                  SwiftUI iOS app for TTFA / RTF / memory / thermal measurements
docs/                             integration notes
scripts/                          xcframework build and golden-vector generation scripts
Models/                           ONNX models (you download them; not in git)
third_party/                      sea-g2p checkout (you clone it; not in git)
```

## Getting started

Three assets are not stored in git and must be prepared once after cloning.

### 1. Build `CSeaG2P.xcframework`

sea-g2p is a Rust library for Vietnamese text normalisation and phonemisation. It is packaged as an xcframework
with slices for macOS (arm64 + x86_64), iOS devices and the iOS simulator.

```bash
git clone https://github.com/pnnbao97/sea-g2p third_party/sea-g2p
git -C third_party/sea-g2p checkout v0.10.0
scripts/build_sea_g2p_xcframework.sh third_party/sea-g2p
```

The result is written to `Packages/SonasiVietnameseKit/Frameworks/CSeaG2P.xcframework`. The build takes a few
minutes and needs several GB of free disk space for cargo's `target/` directory, which you can delete afterwards.

### 2. Download the models

Place the files under `Models/` as follows:

```
Models/
├── onnx_int8/   from Hugging Face: pnnbao-ump/VieNeu-TTS-v3-Turbo (folder onnx_int8)
│                config.json, tokenizer.json, vieneu_prefill.onnx, vieneu_decode_step.onnx,
│                vieneu_acoustic_cached.onnx, vieneu_backbone_shared.data, vieneu_v3_heads.npz
└── codec/       from Hugging Face: OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX
                 moss_audio_tokenizer_decode_step.onnx, its .data file, codec_browser_onnx_meta.json
```

You also need the g2p dictionary `sea_g2p.bin` (about 60 MB), shipped in the sea-g2p repository at
`python/sea_g2p/sea_g2p.bin`. Copy it to `App/SonasiBench/Resources/` to use the iOS benchmark app, or keep it
anywhere and pass its URL to the synthesizer. A sample voice (`voice.npz`: speaker embedding and reference codes)
is included in `App/SonasiBench/Resources/`.

### 3. Run the tests

```bash
cd Packages/SonasiVietnameseKit
swift test
```

The tests compare each stage with the golden vectors in `Tests/Golden`. Strict numerical comparisons use the fp32
model in `Models/onnx_update`; if you only have the int8 model, those tests cannot run. Set `EXPORT_WAVS=1` to
make the end-to-end tests write WAV files you can listen to.

## Usage

The package is meant to be consumed as a local Swift package (or vendored into your app), because the
`CSeaG2P.xcframework` binary target is built locally and is not published. Add
`Packages/SonasiVietnameseKit` as a local package dependency in Xcode or `Package.swift`.

```swift
import SonasiVietnameseKit

let synthesizer = try VieNeuSynthesizer(
    modelDirectory: modelsURL.appendingPathComponent("onnx_int8"),
    codecDirectory: modelsURL.appendingPathComponent("codec"),
    dictionary: Bundle.main.url(forResource: "sea_g2p", withExtension: "bin")!,
    voice: Bundle.main.url(forResource: "voice", withExtension: "npz")!,
    threads: 1,          // threads for frame generation and the codec
    prefillThreads: 2    // threads for the prefill pass only
)

let output = try synthesizer.synthesize(
    "Xin chào! [cười] Hôm nay trời đẹp quá.",
    options: VieNeuSynthesizer.Options(),
    onAudio: { samples in
        // [Float], 48 kHz mono, in playback order, pauses between chunks included
        player.enqueue(samples)
    },
    onChunk: { info in
        print(info.text, info.frames, info.seconds)
    }
)
// output.audio is the whole waveform; output.chunks has per-chunk details.
```

Things to know:

- `synthesize` is synchronous and blocks the calling thread. Call it from a background thread or task.
- Creating a `VieNeuSynthesizer` loads about 400 MB. Create one instance and reuse it.
- sea-g2p needs a thread with a stack of at least 16 MB. The library already runs those calls on a dedicated
  thread; if you call the C API yourself, do the same.
- Code that calls Objective-C APIs (such as ONNX Runtime) in a long loop on a thread without a run loop must wrap
  each iteration in `autoreleasepool`, otherwise memory grows without bound. The library already does this
  internally.

### Options

`VieNeuSynthesizer.Options`:

| Option | Default | Meaning |
|---|---|---|
| `maxChars` | 256 | Maximum characters per chunk |
| `minChunkChars` | 20 | Shorter chunks are merged into a neighbour |
| `firstChunkMaxChars` | 90 | Splits the first chunk to cut time-to-first-audio; raise to 120–150 if the early pause sounds abrupt |
| `sampling` | temperature 0.8, top-k 25, top-p 0.95 | `nil` selects greedy decoding (deterministic, used by tests) |
| `repetitionPenalty` | 1.2 | Penalty applied to recently generated codes |
| `maxNewFrames` | 300 | Upper bound of frames per chunk (one frame is 80 ms) |
| `babbleRetries` | 2 | Regeneration attempts when output looks like babble (only with sampling enabled) |
| `streaming` | `true` | Emit audio inside a chunk as soon as it is decoded |

Pauses between chunks: 0.70 s between paragraphs, 0.50 s between sentences, 0.30 s at commas.

## Benchmarks

### macOS command line

```bash
cd Packages/SonasiVietnameseKit
swift build -c release --product SonasiBench

# SonasiBench <repo-root> <int8|fp32> <threads> [mode]
.build/release/SonasiBench <repo-root> int8 1                    # 19 test sentences
.build/release/SonasiBench <repo-root> int8 1 120                # 120 s of continuous synthesis
.build/release/SonasiBench <repo-root> int8 1 long [prefillThreads]
.build/release/SonasiBench <repo-root> int8 1 longloop           # repeated long text, watch memory
.build/release/SonasiBench <repo-root> int8 1 leak               # memory-leak check
```

Always benchmark Release builds; unoptimised Swift is much slower. The package enables `-O` for Debug as well to
reduce the gap.

### iOS app

```bash
cd App/SonasiBench
xcodegen generate
open SonasiBench.xcodeproj
```

Select your signing team, connect an iPhone and run the Release scheme. The app offers single-sentence runs,
a two-minute sustained run, and long-text synthesis with streaming playback (sample passages of 263, 935 and
2030 characters). It reports TTFA, RTF, memory footprint and thermal state. See
[App/SonasiBench/README.md](App/SonasiBench/README.md) for the full walkthrough and how to read the results.

## Regenerating golden vectors

The golden vectors come from the reference Python implementation (greedy decoding, so they are deterministic).
Run the script inside an environment where `vieneu` and sea-g2p are installed:

```bash
python scripts/gen_golden_vectors.py --out ./golden --precision int8
python scripts/gen_golden_vectors.py --out ./golden --precision fp32
```

Copy the output to `Packages/SonasiVietnameseKit/Tests/Golden/<precision>`. Codebook sequences differ between int8
and fp32, so only compare results of the same precision.

## Technical notes

- **Precision.** int8 is the intended on-device precision. int8 results are not bit-identical across platforms
  (Swift on macOS versus Python on Linux, cosine similarity about 0.996), so the int8 tests check behaviour rather
  than exact codes. fp32 prefill, decoding and codec match the reference to about 1e-6.
- **Acoustic head.** The acoustic graph contains dynamic-quantisation operators and must run with ONNX Runtime
  graph optimisation set to `.none`; other levels diverge from the reference. Prefill, decode step and codec run
  at `.all`.
- **ONNX Runtime.** Version 1.24.2 is the only one available through SwiftPM; only the CPU execution provider
  is used.
- **Audio.** One frame is 80 ms (3840 samples at 48 kHz). The v3 Turbo join logic does not trim chunk edges; it
  only pads zeros to reach the configured pause length.

## Roadmap

- Model downloader with checksum verification.
- Voice catalogue: the library currently loads a single `voice.npz`; the upstream model ships 25 voices.
- Measurements on 4–6 GB iPhones, under concurrent load, and for battery use.
- Distribute the xcframework as a prebuilt binary (release asset or Git LFS) so the package can be added as a
  remote SwiftPM dependency.

## Licence

The source code in this repository is released under the [Apache License 2.0](LICENSE) and is free to use,
modify and distribute, including commercially.

The licence covers this repository's code only. It does not cover the model weights, voices, dictionary or
dependencies you download or link separately, which keep their own terms:

- VieNeu-TTS model weights, voices and sea-g2p by pnnbao97
- MOSS Audio Tokenizer by OpenMOSS
- ONNX Runtime by Microsoft (MIT)

The upstream README and model card of VieNeu-TTS word its usage terms differently (personal use versus commercial
use). Check the current terms of each of these before commercial use or redistribution.

## Acknowledgements

Thanks to the authors of VieNeu-TTS, sea-g2p, the MOSS Audio Tokenizer and ONNX Runtime, whose work this
project is built on.
