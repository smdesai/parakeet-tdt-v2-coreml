#!/usr/bin/env python3
"""Encoder-isolated WER table: NeMo ref vs each CoreML encoder build.

For each build we run CoreML preprocessor+encoder, then decode with NeMo's
RNNT decoder (swap-in methodology). WER is computed vs NeMo's own full-pipeline
transcription (the accuracy ceiling for this conversion).
"""
from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import jiwer
import numpy as np
import soundfile as sf
import torch

from diag_encoder_parity import coreml_mel_and_enc, metrics, rel_l2  # reuse proven helpers

AUDIO_15S = "./audio/yc_first_minute_16k_15s.wav"
NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
FP32_DIR = Path("./parakeet_coreml_v2_fp32")
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")


def decode_encoder(model, enc: np.ndarray, enc_len: np.ndarray) -> str:
    enc_t = torch.from_numpy(np.asarray(enc)).float()
    len_t = torch.from_numpy(np.asarray(enc_len)).long().reshape(-1)
    with torch.no_grad():
        preds = model.decoding.rnnt_decoder_predictions_tensor(enc_t, len_t, return_hypotheses=True)
    h = preds[0] if isinstance(preds, tuple) else preds
    if isinstance(h, list):
        h = h[0]
    return h.text if hasattr(h, "text") else str(h)


def norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def main():
    import nemo.collections.asr as nemo_asr

    audio, sr = sf.read(AUDIO_15S)
    audio = audio.astype(np.float32)

    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    # NeMo encoder reference FIRST — transcribe() mutates preprocessor state
    # (dither/pad_to), which would corrupt a manual encode done afterward.
    with torch.no_grad():
        mel_t, mel_len_t = m.preprocessor(
            input_signal=torch.tensor(audio).unsqueeze(0).float(),
            length=torch.tensor([len(audio)]),
        )
        enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
    nemo_enc = enc_t.cpu().numpy()
    nemo_enc_len = enc_len_t.cpu().numpy()

    # NeMo full-pipeline reference transcription
    ref = m.transcribe([AUDIO_15S], batch_size=1)[0]
    ref_text = ref.text if hasattr(ref, "text") else str(ref)
    print(f"\nNeMo reference: {ref_text!r}\n")

    # Sanity: decode NeMo's own encoder output (should match transcribe())
    nemo_swap_text = decode_encoder(m, nemo_enc, nemo_enc_len)

    rows = [{
        "variant": "NeMo PyTorch (reference)",
        "precision": "FP32 (torch)",
        "encoder_rel_l2": 0.0,
        "transcription": ref_text,
        "wer_vs_ref": 0.0,
        "swap_decode_text": nemo_swap_text,
    }]

    builds = [
        ("New CoreML FP32", "FLOAT32", FP32_DIR / "parakeet_preprocessor.mlpackage", FP32_DIR / "parakeet_encoder.mlpackage"),
        ("New CoreML FP16", "FLOAT16", FP16_DIR / "parakeet_preprocessor.mlpackage", FP16_DIR / "parakeet_encoder.mlpackage"),
        ("Shipped baseline", "6-bit palettized", BASE_DIR / "Preprocessor.mlmodelc", BASE_DIR / "Encoder.mlmodelc"),
    ]

    for name, prec, prep, enc in builds:
        if not (prep.exists() and enc.exists()):
            print(f"[{name}] missing, skip")
            continue
        try:
            _, cml_enc, cml_enc_len = coreml_mel_and_enc(prep, enc, audio)
            if cml_enc_len is None:
                cml_enc_len = np.array([cml_enc.shape[2]], dtype=np.int32)
            text = decode_encoder(m, cml_enc, cml_enc_len)
            r = rel_l2(nemo_enc, cml_enc) if cml_enc.shape == nemo_enc.shape else float("nan")
            wer = jiwer.wer(norm(ref_text), norm(text))
            rows.append({
                "variant": name,
                "precision": prec,
                "encoder_rel_l2": r,
                "transcription": text,
                "wer_vs_ref": wer,
            })
            print(f"[{name}] rel_l2={r:.3e}  WER={wer:.4f}")
            print(f"   text: {text!r}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"variant": name, "precision": prec, "error": repr(e)})
            print(f"[{name}] ERROR {e!r}")

    out = {"audio": AUDIO_15S, "reference": ref_text, "rows": rows}
    Path("./bench_results").mkdir(exist_ok=True)
    Path("./bench_results/v2_wer_table.json").write_text(json.dumps(out, indent=2))

    # pretty table
    print("\n=== Encoder-isolated WER (vs NeMo PyTorch reference) ===")
    print(f"{'variant':28} {'precision':18} {'enc_rel_l2':>12} {'WER':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['variant']:28} {r['precision']:18} {'ERROR':>12} {'-':>8}")
        else:
            print(f"{r['variant']:28} {r['precision']:18} {r['encoder_rel_l2']:>12.3e} {r['wer_vs_ref']:>8.4f}")
    print("\nwrote ./bench_results/v2_wer_table.json")


if __name__ == "__main__":
    main()
