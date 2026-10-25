"""Phase 0 — golden vectors + chữ ký graph cho SonasiVietnameseKit.

Chạy TRONG container vieneu-tts-api-cpu (đã có model cache + sea-g2p + vieneu):

    docker cp gen_golden_vectors.py vieneu-tts-api-cpu:/tmp/
    docker exec -e OMP_NUM_THREADS=1 vieneu-tts-api-cpu \
        python /tmp/gen_golden_vectors.py --out /tmp/golden --precision int8
    docker cp vieneu-tts-api-cpu:/tmp/golden ./Packages/SonasiVietnameseKit/Tests/Golden

Sinh greedy (temperature=0 → argmax, repetition penalty vẫn áp dụng, tất định),
nên bản Swift phải ra ĐÚNG cùng mã codebook. Mỗi câu một file .npz nhỏ (< 300 KB).
Chưa được chạy thử — script viết dựa trên source onnx_runtime_lite.py (commit 04044e1).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SENTENCES = [
    "Xin chào, tôi là trợ lý dịch thuật của Sonasi.",
    "Cuộc họp bắt đầu lúc 9h30 ngày 24/10/2026 tại tầng 3.",
    "Doanh thu quý 3 tăng 12% so với cùng kỳ, đạt 1,5 tỷ đồng.",
    "Chúng tôi sẽ deploy bản release mới lên server vào thứ Hai.",
    "Giá SP500 hôm nay là 4.200,5 điểm.",
    "Thế à?",
    "Vâng.",
    "Anh có thể gửi lại file báo cáo trước năm giờ chiều được không?",
    "Nhà máy ở Bắc Ninh đang chạy ba ca, mỗi ca tám tiếng.",
    "Hệ thống MES ghi nhận tỉ lệ lỗi giảm còn 0,8 phần trăm.",
    "Đề nghị mọi người tắt micro khi không phát biểu.",
    "Theo kế hoạch, giai đoạn một sẽ hoàn thành vào cuối tháng mười hai.",
    "Khách hàng Nhật Bản yêu cầu báo giá bằng yên và đô la Mỹ.",
    "Tôi đồng ý với ý kiến của chị Lan [cười] nhưng cần thêm số liệu.",
    "Email liên hệ là support@sonasi.app, số điện thoại 0912 345 678.",
    "Một, hai, ba, bốn, năm.",
    "Mô hình chạy hoàn toàn trên thiết bị, không cần kết nối mạng.",
    "Xin lỗi, anh nói chậm lại một chút được không ạ?",
    "Tổng chi phí dự kiến khoảng 2,35 triệu đô, đã bao gồm VAT 10%.",
    "Cảm ơn mọi người đã tham dự buổi họp hôm nay. Hẹn gặp lại tuần sau!",
]


def sig(sess):
    f = lambda xs: [{"name": x.name, "type": x.type, "shape": [d if isinstance(d, int) else str(d) for d in x.shape]} for x in xs]
    return {"inputs": f(sess.get_inputs()), "outputs": f(sess.get_outputs())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/golden")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--voice", default="Hải Đăng")
    ap.add_argument("--wav-samples", type=int, default=9600, help="số mẫu 48 kHz đầu tiên lưu lại (0.2 s)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions, normalize_to_chunks_v3_with_gaps
    from vieneu_utils.core_utils import max_expected_frames
    from vieneu._v3_turbo_engine.rep_history import RepetitionHistory, DEFAULT_REP_WINDOW
    import vieneu

    eng = OnnxV3LiteEngine(onnx_subfolder="onnx_int8" if a.precision == "int8" else "onnx_update", threads=1)

    # 1) Chữ ký đầy đủ của mọi graph (kể cả 54 in/out của codec decode_step)
    sigs = {"prefill": sig(eng.sess_pre), "decode_step": sig(eng.sess_dec), "acoustic_cached": sig(eng.sess_ac),
            "codec_decode_full": sig(eng.sess_codec_dec)}
    if eng.sess_codec_step is not None:
        sigs["codec_decode_step"] = sig(eng.sess_codec_step)
    (out / "graph_signatures.json").write_text(json.dumps(sigs, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "codec_streaming_spec.json").write_text(json.dumps(eng._codec_stream_spec, indent=1), encoding="utf-8")

    # 2) Giọng
    vj = json.loads((Path(vieneu.__file__).parent / "assets" / "voices_v3_turbo.json").read_text(encoding="utf-8"))
    v = vj["presets"][a.voice] if "presets" in vj else vj[a.voice]
    emb = np.asarray(v["speaker_emb"], np.float32)
    ref = np.asarray(v["codes"], np.int64)
    anchor = eng._speaker_anchor(emb)
    np.savez(out / "voice.npz", speaker_emb=emb, ref_codes=ref, anchor=anchor)

    # 3) Chunking của một đoạn dài (để test port core_utils)
    long_text = " ".join(SENTENCES[:8]) + "\n" + " ".join(SENTENCES[8:])
    chunks, gaps = normalize_to_chunks_v3_with_gaps(long_text)
    (out / "chunking.json").write_text(json.dumps({"text": long_text, "chunks": chunks, "gaps": gaps}, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = []
    for i, text in enumerate(SENTENCES):
        ph = phonemize_text_with_emotions(text)
        ids = eng.tokenizer.encode(ph, add_special_tokens=False).ids
        rows = eng._build_rows(ph, ref, eng._resolve_style_id())
        emb_in = eng._embed_rows(rows, anchor)                     # (1,T,768)
        cap = min(300, max_expected_frames(ph))

        pre = eng.sess_pre.run(None, {"inputs_embeds": emb_in})
        past_k = [pre[1 + j] for j in range(eng.L)]
        past_v = [pre[1 + eng.L + j] for j in range(eng.L)]
        h = pre[0][:, -1]
        prefill_last = h.copy()
        T = emb_in.shape[1]
        hist = RepetitionHistory(eng.n_vq, DEFAULT_REP_WINDOW)
        frames, eos_at = [], None
        for t in range(cap):
            codes, eos = eng._acoustic_frame(h, 0.0, 25, 0.95, 1.2, hist)   # greedy
            frames.append(codes)
            if eos:
                eos_at = t; break
            slot = np.full((1, eng.n_vq + 1), eng.audio_pad, np.int64); slot[0, 0] = eng.sgs; slot[0, 1:] = codes
            feed = {"inputs_embeds": eng._embed_rows(slot, anchor), "position_ids": np.array([[T + t]], np.int64)}
            for j in range(eng.L):
                feed[f"past_k_{j}"] = past_k[j]; feed[f"past_v_{j}"] = past_v[j]
            o = eng.sess_dec.run(None, feed)
            h = o[0][:, 0]
            past_k = [o[1 + j] for j in range(eng.L)]; past_v = [o[1 + eng.L + j] for j in range(eng.L)]
        frames = np.asarray(frames, np.int32)
        wav = eng._decode_codes(frames) if len(frames) else np.zeros(0, np.float32)
        np.savez_compressed(
            out / f"s{i:02d}.npz",
            text=text, phonemes=ph, token_ids=np.asarray(ids, np.int32), rows=rows.astype(np.int32),
            embeds_head=emb_in[0, :4].astype(np.float32), embeds_tail=emb_in[0, -2:].astype(np.float32),
            embeds_sum=np.float64(emb_in.astype(np.float64).sum()),
            prefill_last_hidden=prefill_last.astype(np.float32), frames=frames,
            wav_head=wav[: a.wav_samples].astype(np.float32), wav_len=np.int64(wav.size),
            wav_rms=np.float64(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0,
            frame_cap=np.int32(cap), eos_at=np.int32(-1 if eos_at is None else eos_at))
        manifest.append({"i": i, "text": text, "phonemes": ph, "n_tokens": len(ids), "n_frames": int(len(frames)),
                         "eos": eos_at is not None, "codes_sha1": hashlib.sha1(frames.tobytes()).hexdigest()})
        print(i, len(ids), len(frames), "eos" if eos_at is not None else "CAP", ph[:60])
    (out / "manifest.json").write_text(json.dumps({"precision": a.precision, "voice": a.voice, "items": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
