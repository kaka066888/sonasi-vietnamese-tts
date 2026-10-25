import Foundation

/// Văn bản dài → audio 48 kHz mono: chunk (v3) → phoneme → sinh (trần frame + babble guard) → codec → nối có khoảng nghỉ.
public final class VieNeuSynthesizer: @unchecked Sendable {
    public struct Options: Sendable {
        public var maxChars = 256, minChunkChars = 20
        /// Chunk đầu tối đa N ký tự (0 = tắt): prompt prefill ngắn hơn → TTFA thấp hơn, đổi lại thêm một ranh giới ở đầu bài.
        public var firstChunkMaxChars = 90
        public var sampling: VieNeuGenerator.Sampling? = .init()      // nil = greedy
        public var repetitionPenalty: Float = 1.2
        public var maxNewFrames = 300
        public var babbleRetries = 2
        /// Phát audio theo nhóm frame ngay trong chunk (chỉ với chunk mà babble guard không áp dụng, nên không mất kiểm tra).
        public var streaming = true
        public init() {}
    }

    public struct ChunkInfo: Sendable {
        public let text: String, phonemes: String
        public let frames: Int, capFrames: Int, retries: Int, babbleSuspect: Bool
        public let seconds: Double          // thời gian tính cho chunk (g2p + sinh + codec)
    }

    public struct Output: Sendable {
        public let audio: [Float]           // 48 kHz mono, đã gồm khoảng nghỉ giữa chunk
        public let chunks: [ChunkInfo]
        public var seconds: Double { Double(audio.count) / Double(VieNeuAudio.sampleRate) }
    }

    /// Chẩn đoán TTFA: nhận (sự kiện, giây kể từ lúc gọi synthesize) cho chunk đầu tiên.
    public var trace: (@Sendable (String, Double) -> Void)?
    private var traceStart: UInt64 = 0
    private func mark(_ event: String) {
        trace?(event, Double(DispatchTime.now().uptimeNanoseconds - traceStart) / 1e9)
    }

    private let tokenizer: VieNeuTokenizer
    private let embedder: VieNeuEmbedder
    private let generator: VieNeuGenerator
    private let codec: VieNeuCodecStream
    private let g2p: SeaG2P
    private let refRows: [[Int]]
    private let anchor: [Float]

    public init(modelDirectory: URL, codecDirectory: URL, dictionary: URL, voice: URL, threads: Int = 1, prefillThreads: Int? = nil) throws {
        tokenizer = try VieNeuTokenizer(tokenizerJSON: Data(contentsOf: modelDirectory.appendingPathComponent("tokenizer.json")))
        let cfg = try VieNeuEmbedder.Config(configJSON: Data(contentsOf: modelDirectory.appendingPathComponent("config.json")))
        embedder = try VieNeuEmbedder(config: cfg, headsURL: modelDirectory.appendingPathComponent("vieneu_v3_heads.npz"))
        generator = try VieNeuGenerator(embedder: embedder, modelDirectory: modelDirectory, threads: threads, prefillThreads: prefillThreads)
        codec = try VieNeuCodecStream(directory: codecDirectory, threads: threads)
        g2p = try SeaG2P(dictionaryPath: dictionary.path)
        let v = try NPZ.read(voice)
        let ref = try v["ref_codes"]!.ints(), n = cfg.nVQ
        refRows = stride(from: 0, to: ref.count, by: n).map { Array(ref[$0..<($0 + n)]) }
        anchor = embedder.speakerAnchor(try v["speaker_emb"]!.floats())
    }

    /// Chỉ tách chunk (không sinh audio) — để đo chi phí lập kế hoạch trước khi có audio đầu tiên.
    public func plan(_ text: String, options: Options = .init()) throws -> VieNeuChunker.Result {
        let plan = try VieNeuChunker.chunks(for: text, g2p: g2p, maxChars: options.maxChars, minChunkChars: options.minChunkChars)
        return try VieNeuChunker.splitFirst(plan, maxChars: options.firstChunkMaxChars)
    }

    /// `onAudio` nhận từng phần audio theo thứ tự (zeros nghỉ + mẫu mới) ngay khi có; nối lại = `Output.audio`.
    /// `onChunk` gọi khi một chunk xong.
    public func synthesize(_ text: String, options: Options = .init(),
                           onAudio: (([Float]) throws -> Void)? = nil,
                           onChunk: ((ChunkInfo) throws -> Void)? = nil) throws -> Output {
        traceStart = DispatchTime.now().uptimeNanoseconds
        let plan = try self.plan(text, options: options)
        mark("chunk plan")
        let silence = VieNeuAudio.silence(for: plan.gaps)
        var infos: [ChunkInfo] = [], out: [Float] = [], prevWav: [Float] = []
        for (i, chunk) in plan.chunks.enumerated() {
            var started = false, wavSoFar: [Float] = []
            func emit(_ piece: [Float]) throws {
                if !started {
                    started = true
                    if i > 0 {
                        // Khoảng nghỉ đo bằng đuôi chunk trước + đầu của phần audio ĐẦU TIÊN (chunk stream: gần đúng).
                        let pad = VieNeuAudio.pausePadSamples(prev: prevWav, next: piece, sr: VieNeuAudio.sampleRate, pauseSeconds: silence[i - 1])
                        if pad > 0 { let z = [Float](repeating: 0, count: pad); out += z; try onAudio?(z) }
                    }
                }
                wavSoFar += piece; out += piece
                try onAudio?(piece)
            }
            let isFirst = i == 0
            let info = try autoreleasepool { try synthesizeChunk(chunk, options, traceThis: isFirst, emit: emit) }
            prevWav = wavSoFar
            infos.append(info)
            try onChunk?(info)
        }
        return Output(audio: out, chunks: infos)
    }

    private func synthesizeChunk(_ text: String, _ o: Options, traceThis: Bool, emit: @escaping ([Float]) throws -> Void) throws -> ChunkInfo {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let phonemes = try VieNeuEmotion.phonemize(text, g2p)
        if traceThis { mark("phoneme hóa chunk 1") }
        let cap = min(o.maxNewFrames, VieNeuFrameCap.maxExpectedFrames(phonemes))
        let rows = embedder.buildRows(phonemeIDs: tokenizer.encode(phonemes), refCodes: refRows)
        if traceThis { mark("tokenize (\(rows.count) hàng prompt)") }

        let syl = VieNeuFrameCap.syllableCount(phonemes)
        let canStream = o.streaming && !VieNeuFrameCap.isCueOnly(phonemes)
            && (syl == 0 || syl > VieNeuAudio.babbleMaxSyllables || phonemes.contains("<|emotion_"))
        if canStream {
            try codec.reset()
            var pending: [[Int]] = [], flushed = false, total = 0
            func flush() throws {
                guard !pending.isEmpty else { return }
                if traceThis && !flushed { mark("2 frame đầu xong") }
                let w = try codec.decode(pending); pending = []; total += w.count
                if traceThis && !flushed { mark("codec giải mã xong phần đầu") }
                flushed = true
                try emit(w)
            }
            let r = try generator.generate(rows: rows, anchor: anchor, frameCap: cap, repPenalty: o.repetitionPenalty, sampling: o.sampling,
                                           onPrefillDone: { if traceThis { self.mark("prefill xong") } },
                                           onFrame: { _, codes, last in
                pending.append(codes)
                if pending.count >= (flushed ? 4 : 2) || last { try flush() }
            })
            let secs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            return ChunkInfo(text: text, phonemes: phonemes, frames: r.frames.count, capFrames: cap, retries: 0, babbleSuspect: false, seconds: secs)
        }

        func once() throws -> (wav: [Float], frames: Int) {
            let r = try generator.generate(rows: rows, anchor: anchor, frameCap: cap, repPenalty: o.repetitionPenalty, sampling: o.sampling)
            guard !r.frames.isEmpty else { return ([], 0) }
            try codec.reset()
            return (try codec.decode(r.frames), r.frames.count)
        }

        var best = try once()
        var verdict = VieNeuAudio.babbleSuspect(best.wav, sr: VieNeuAudio.sampleRate, phonemes: phonemes, capFrames: cap, frames: best.frames)
        var tries = 0
        // Sinh lại chỉ có ích khi có lấy mẫu ngẫu nhiên; greedy sẽ ra y hệt.
        while !best.wav.isEmpty, verdict.suspect, tries < o.babbleRetries, o.sampling != nil {
            tries += 1
            let cand = try once()
            guard !cand.wav.isEmpty else { continue }
            let v = VieNeuAudio.babbleSuspect(cand.wav, sr: VieNeuAudio.sampleRate, phonemes: phonemes, capFrames: cap, frames: cand.frames)
            if VieNeuAudio.babblePrefer(v, over: verdict) { best = cand; verdict = v }
        }
        if !best.wav.isEmpty { try emit(best.wav) }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        return ChunkInfo(text: text, phonemes: phonemes, frames: best.frames, capFrames: cap, retries: tries, babbleSuspect: verdict.suspect, seconds: secs)
    }
}
