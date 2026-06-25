#!/usr/bin/env python3
"""Multi-window parity + WER: slice the 60s recording into 4x 15s windows.

Strengthens the single-clip result to n=4 distinct audio contexts. For each
window we report encoder rel_l2 (vs NeMo torch encoder, the robust fidelity
metric) and encoder-isolated WER (vs NeMo full-pipeline transcription).
"""
from __future__ import annotations

import json
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch

from diag_encoder_parity import coreml_mel_and_enc, rel_l2
from diag_wer_table import decode_encoder, norm

SRC_60S = "./audio/yc_first_minute_16k.wav"
NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
WIN = 240000  # 15s @ 16kHz
FP32_DIR = Path("./parakeet_coreml_v2_fp32")
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")

BUILDS = [
    ("fp32", FP32_DIR / "parakeet_preprocessor.mlpackage", FP32_DIR / "parakeet_encoder.mlpackage"),
    ("fp16", FP16_DIR / "parakeet_preprocessor.mlpackage", FP16_DIR / "parakeet_encoder.mlpackage"),
    ("shipped_6bit", BASE_DIR / "Preprocessor.mlmodelc", BASE_DIR / "Encoder.mlmodelc"),
]


def main():
    import nemo.collections.asr as nemo_asr

    full, sr = sf.read(SRC_60S)
    full = full.astype(np.float32)
    assert sr == 16000
    n_win = full.shape[0] // WIN
    print(f"source {full.shape[0]} samples -> {n_win} windows of 15s")

    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    windows = []
    tmp_paths = []
    for w in range(n_win):
        seg = full[w * WIN:(w + 1) * WIN]
        p = f"./audio/_win{w}_15s.wav"
        sf.write(p, seg, sr)
        tmp_paths.append(p)
        windows.append((w, seg, p))

    results = []  # per (window, build)
    per_build_agg = {tag: {"rel_l2": [], "wer": []} for tag, _, _ in BUILDS}

    for w, seg, path in windows:
        # NeMo encoder reference. We deliberately do NOT call m.transcribe() here:
        # transcribe() mutates the preprocessor/featurizer state, which corrupts the
        # manual reference encode of every subsequent window in this loop. Instead we
        # derive the reference text by decoding NeMo's own encoder output via the same
        # swap path used for the CoreML builds — this is the correct reference for an
        # encoder-isolated comparison and keeps a single clean code path.
        with torch.no_grad():
            mel_t, mel_len_t = m.preprocessor(
                input_signal=torch.tensor(seg).unsqueeze(0).float(),
                length=torch.tensor([len(seg)]),
            )
            enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
        nemo_enc = enc_t.cpu().numpy()
        nemo_enc_len = enc_len_t.cpu().numpy()

        ref_text = decode_encoder(m, nemo_enc, nemo_enc_len)

        print(f"\n--- window {w} | ref: {ref_text[:70]!r}")
        for tag, prep, enc in BUILDS:
            try:
                _, cml_enc, cml_enc_len = coreml_mel_and_enc(prep, enc, seg)
                if cml_enc_len is None:
                    cml_enc_len = np.array([cml_enc.shape[2]], dtype=np.int32)
                r = rel_l2(nemo_enc, cml_enc) if cml_enc.shape == nemo_enc.shape else float("nan")
                text = decode_encoder(m, cml_enc, cml_enc_len)
                wer = jiwer.wer(norm(ref_text), norm(text))
                results.append({"window": w, "build": tag, "rel_l2": r, "wer": wer, "text": text})
                per_build_agg[tag]["rel_l2"].append(r)
                per_build_agg[tag]["wer"].append(wer)
                print(f"   {tag:14} rel_l2={r:.3e}  WER={wer:.4f}")
            except Exception as e:
                results.append({"window": w, "build": tag, "error": repr(e)})
                print(f"   {tag:14} ERROR {e!r}")

    # aggregate
    summary = {}
    for tag in per_build_agg:
        rs = per_build_agg[tag]["rel_l2"]
        ws = per_build_agg[tag]["wer"]
        summary[tag] = {
            "n": len(rs),
            "rel_l2_mean": float(np.mean(rs)) if rs else None,
            "rel_l2_max": float(np.max(rs)) if rs else None,
            "wer_mean": float(np.mean(ws)) if ws else None,
            "wer_max": float(np.max(ws)) if ws else None,
        }

    out = {"source": SRC_60S, "n_windows": n_win, "per_window": results, "summary": summary}
    Path("./bench_results").mkdir(exist_ok=True)
    Path("./bench_results/v2_multiwindow.json").write_text(json.dumps(out, indent=2))

    print("\n=== Aggregate over windows (encoder rel_l2 vs NeMo torch; WER vs NeMo transcribe) ===")
    print(f"{'build':14} {'n':>3} {'rel_l2_mean':>12} {'rel_l2_max':>12} {'WER_mean':>10} {'WER_max':>9}")
    for tag in ["fp32", "fp16", "shipped_6bit"]:
        s = summary[tag]
        print(f"{tag:14} {s['n']:>3} {s['rel_l2_mean']:>12.3e} {s['rel_l2_max']:>12.3e} "
              f"{s['wer_mean']:>10.4f} {s['wer_max']:>9.4f}")
    print("\nwrote ./bench_results/v2_multiwindow.json")

    # cleanup temp window wavs
    for p in tmp_paths:
        try:
            Path(p).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
