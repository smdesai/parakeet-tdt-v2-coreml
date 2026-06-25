#!/usr/bin/env python3
"""Gate compressed encoder candidates against FP16 + shipped 6-bit.

Reuses the trusted diag_dataset_wer machinery (CoreMLEncoder with true_len
threading, NeMo-decode swap, identical chunking). For each candidate build dir
it reports:
  - encoder rel_l2 vs NeMo (single 15s clip, the robust fidelity metric)
  - WER vs ground truth, drift-vs-NeMo, exact-NeMo-match over a dataset subset

Pass/fail rule: a candidate PASSES if its drift-from-NeMo stays near FP16's
(~0.003) rather than near the shipped 6-bit's (~0.076). rel_l2 must be << 0.238
(the shipped baseline's encoder rel_l2).

Usage:
  gate_compressed.py --builds int8,palett8 --dataset fda --limit 80
  gate_compressed.py --builds int8 --dataset earnings22 --limit 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import jiwer
import numpy as np
import torch

from diag_dataset_wer import (
    CoreMLEncoder, DATASETS, SR, load_16k_mono, norm,
    transcribe_chunked, windows_of,
)
from diag_encoder_parity import rel_l2

NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")
PARITY_CLIP = "./audio/yc_first_minute_16k_15s.wav"

# build registry: name -> (preprocessor, encoder)
KNOWN = {
    "fp16": (FP16_DIR / "parakeet_preprocessor.mlpackage", FP16_DIR / "parakeet_encoder.mlpackage"),
    "shipped_6bit": (BASE_DIR / "Preprocessor.mlmodelc", BASE_DIR / "Encoder.mlmodelc"),
    "int8": (Path("./parakeet_coreml_v2_int8/parakeet_preprocessor.mlpackage"),
             Path("./parakeet_coreml_v2_int8/parakeet_encoder.mlpackage")),
    "palett8": (Path("./parakeet_coreml_v2_palett8/parakeet_preprocessor.mlpackage"),
                Path("./parakeet_coreml_v2_palett8/parakeet_encoder.mlpackage")),
    "palett6": (Path("./parakeet_coreml_v2_palett6/parakeet_preprocessor.mlpackage"),
                Path("./parakeet_coreml_v2_palett6/parakeet_encoder.mlpackage")),
    "palett4": (Path("./parakeet_coreml_v2_palett4/parakeet_preprocessor.mlpackage"),
                Path("./parakeet_coreml_v2_palett4/parakeet_encoder.mlpackage")),
}


def dir_size_gb(p: Path) -> float:
    total = 0
    for root, _d, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default="fp16,int8,shipped_6bit")
    ap.add_argument("--dataset", choices=list(DATASETS), default="fda")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="./bench_results/v2_compressed_gate.json")
    args = ap.parse_args()

    builds = args.builds.split(",")
    import nemo.collections.asr as nemo_asr
    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    # --- NeMo encoder fn (reference) ---
    def nemo_encoder_fn(window, true_len):
        with torch.no_grad():
            mel_t, mel_len_t = m.preprocessor(
                input_signal=torch.tensor(window).unsqueeze(0).float(),
                length=torch.tensor([true_len]),
            )
            enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
        return enc_t.cpu().numpy(), enc_len_t.cpu().numpy()

    # --- load candidate encoders ---
    encs = {}
    sizes = {}
    for b in builds:
        prep, enc = KNOWN[b]
        if not (prep.exists() and enc.exists()):
            print(f"[{b}] MISSING ({enc}); skipping")
            continue
        encs[b] = CoreMLEncoder(prep, enc)
        sizes[b] = round(dir_size_gb(enc.parent if enc.suffix == ".mlpackage" else enc), 3)

    # --- rel_l2 on the parity clip ---
    print("\n=== encoder rel_l2 vs NeMo (single 15s clip) ===")
    import soundfile as sf
    wav, sr = sf.read(PARITY_CLIP)
    wav = wav.astype(np.float32)
    assert sr == SR
    seg = wav[:240000]
    if seg.shape[0] < 240000:
        seg = np.pad(seg, (0, 240000 - seg.shape[0]))
    true_len = min(wav.shape[0], 240000)
    nemo_enc, _ = nemo_encoder_fn(seg, true_len)
    rel = {}
    for b in encs:
        ce, _ = encs[b].encode(seg, true_len)
        r = rel_l2(nemo_enc, ce) if ce.shape == nemo_enc.shape else float("nan")
        rel[b] = r
        print(f"  {b:14} rel_l2={r:.4e}   size={sizes[b]}GB")

    # --- WER over dataset subset ---
    root = DATASETS[args.dataset]
    wavs = sorted(glob.glob(os.path.join(root, "*.wav")))[args.offset:]
    if args.limit:
        wavs = wavs[:args.limit]
    print(f"\n=== WER on {args.dataset} ({len(wavs)} clips) ===")

    agg = {b: {"wt": [], "vn": [], "exact": 0} for b in encs}
    agg["nemo"] = {"wt": []}
    per_clip = []
    n_used = 0
    for i, wpath in enumerate(wavs):
        tpath = wpath[:-4]
        gt_file = next((c for c in (tpath + ".text.txt", tpath + ".txt") if os.path.exists(c)), None)
        if not gt_file:
            continue
        gt = norm(Path(gt_file).read_text())
        if not gt.strip():
            continue
        wav = load_16k_mono(wpath)
        n_used += 1

        nemo_txt = transcribe_chunked(nemo_encoder_fn, m, wav)
        agg["nemo"]["wt"].append(jiwer.wer(gt, norm(nemo_txt)))
        row = {"clip": os.path.basename(wpath)}
        for b in encs:
            txt = transcribe_chunked(lambda w, tl, e=encs[b]: e.encode(w, tl), m, wav)
            wt = jiwer.wer(gt, norm(txt))
            vn = jiwer.wer(norm(nemo_txt), norm(txt))
            agg[b]["wt"].append(wt)
            agg[b]["vn"].append(vn)
            if norm(txt) == norm(nemo_txt):
                agg[b]["exact"] += 1
            row[f"{b}_wt"] = round(wt, 4)
            row[f"{b}_vn"] = round(vn, 4)
        per_clip.append(row)
        if (i + 1) % 20 == 0:
            print(f"  …{i+1}/{len(wavs)}")

    # --- summary ---
    summary = {"dataset": args.dataset, "n": n_used, "rel_l2": rel, "sizes_gb": sizes, "builds": {}}
    nemo_wt = float(np.mean(agg["nemo"]["wt"]))
    summary["nemo_wer_truth"] = nemo_wt
    print(f"\n=== SUMMARY  dataset={args.dataset}  n={n_used} ===")
    print(f"  nemo            WER_truth={nemo_wt:.4f}")
    print(f"  {'build':14} {'size_GB':>7} {'rel_l2':>10} {'WER_truth':>10} {'drift_NeMo':>11} {'exact%':>7}")
    for b in encs:
        wt = float(np.mean(agg[b]["wt"]))
        vn = float(np.mean(agg[b]["vn"]))
        ex = agg[b]["exact"] / n_used if n_used else float("nan")
        summary["builds"][b] = {"size_gb": sizes[b], "rel_l2": rel[b],
                                "wer_truth": wt, "drift_nemo": vn, "exact_match": ex}
        print(f"  {b:14} {sizes[b]:>7.2f} {rel[b]:>10.3e} {wt:>10.4f} {vn:>11.4f} {ex:>6.0%}")

    out = {"summary": summary, "per_clip": per_clip}
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
