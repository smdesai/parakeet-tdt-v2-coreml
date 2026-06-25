#!/usr/bin/env python3
"""Gate the INT8 'final' encoder on LibriSpeech test-clean — the dataset the
FluidAudio `asr-benchmark` command uses (HuggingFace FluidInference/librispeech,
downloaded to ~/Library/Application Support/FluidAudio/Datasets/LibriSpeech).

Same trusted encoder-isolated methodology as gate_compressed.py: CoreML
preprocessor+encoder -> NeMo RNNT decode (swap-in), identical 15s chunking and
true_len threading across all systems, so WER differences are purely encoder.

Reports per system: WER vs ground truth, drift-vs-NeMo, exact-NeMo-match.

Usage:
  gate_librispeech.py --builds final,shipped_6bit --limit 0   # 0 = all 2620
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jiwer
import numpy as np
import torch

from diag_dataset_wer import CoreMLEncoder, SR, load_16k_mono, norm, transcribe_chunked

NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
# Deployable encoder in .mlpackage form (INT8 encoder + FP16 preprocessor). The
# compiled .mlmodelc runtime dir (parakeet_coreml_v2_final) carries no .mlpackage,
# so this points at the INT8 .mlpackage pair (== INT8_DIR below).
FINAL_DIR = Path("./parakeet_coreml_v2_int8")
INT8_DIR = Path("./parakeet_coreml_v2_int8")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")
# NOTE: the live BASE_DIR was overwritten with the INT8 build on 2026-06-19 16:59
# (an external install backed the genuine 6-bit baseline up to *-v2-orig first).
# Use shipped_6bit_orig for a VALID 6-bit comparison; "shipped_6bit" now == int8.
ORIG_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-orig")
LS_ROOT = Path("/Users/sdesai/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean")

KNOWN = {
    "final": (FINAL_DIR / "parakeet_preprocessor.mlpackage", FINAL_DIR / "parakeet_encoder.mlpackage"),
    "int8": (INT8_DIR / "parakeet_preprocessor.mlpackage", INT8_DIR / "parakeet_encoder.mlpackage"),
    "fp16": (FP16_DIR / "parakeet_preprocessor.mlpackage", FP16_DIR / "parakeet_encoder.mlpackage"),
    "shipped_6bit": (BASE_DIR / "Preprocessor.mlmodelc", BASE_DIR / "Encoder.mlmodelc"),  # CAUTION: live dir == int8 since 16:59
    "shipped_6bit_orig": (ORIG_DIR / "Preprocessor.mlmodelc", ORIG_DIR / "Encoder.mlmodelc"),  # genuine 6-bit palettized
}


def collect_librispeech(root: Path):
    """Yield (flac_path, ground_truth_text) from LibriSpeech .trans.txt files."""
    pairs = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            flac = trans.parent / f"{uid}.flac"
            if flac.exists():
                pairs.append((flac, text))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", default="final,shipped_6bit")
    ap.add_argument("--limit", type=int, default=0, help="0 = all utterances")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="./bench_results/v2_librispeech_final.json")
    args = ap.parse_args()

    builds = args.builds.split(",")
    import nemo.collections.asr as nemo_asr
    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    def nemo_encoder_fn(window, true_len):
        with torch.no_grad():
            mel_t, mel_len_t = m.preprocessor(
                input_signal=torch.tensor(window).unsqueeze(0).float(),
                length=torch.tensor([true_len]),
            )
            enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
        return enc_t.cpu().numpy(), enc_len_t.cpu().numpy()

    encs = {}
    for b in builds:
        prep, enc = KNOWN[b]
        if not (prep.exists() and enc.exists()):
            print(f"[{b}] MISSING ({enc}); skipping")
            continue
        encs[b] = CoreMLEncoder(prep, enc)

    pairs = collect_librispeech(LS_ROOT)
    print(f"LibriSpeech test-clean: {len(pairs)} utterances")
    pairs = pairs[args.offset:]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"processing {len(pairs)} (offset={args.offset})")

    agg = {b: {"wt": [], "vn": [], "exact": 0} for b in encs}
    agg["nemo"] = {"wt": []}
    per_clip = []
    n = 0
    for i, (flac, gt_raw) in enumerate(pairs):
        gt = norm(gt_raw)
        if not gt.strip():
            continue
        wav = load_16k_mono(str(flac))
        n += 1

        nemo_txt = transcribe_chunked(nemo_encoder_fn, m, wav)
        agg["nemo"]["wt"].append(jiwer.wer(gt, norm(nemo_txt)))
        row = {"clip": flac.stem, "dur_s": round(wav.shape[0] / SR, 1)}
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
        if (i + 1) % 100 == 0:
            cur = "  ".join(f"{b}={np.mean(agg[b]['wt']):.4f}" for b in encs)
            print(f"  [{i+1}/{len(pairs)}] nemo={np.mean(agg['nemo']['wt']):.4f}  {cur}")

    summary = {"dataset": "librispeech-test-clean", "n": n,
               "nemo_wer_truth": float(np.mean(agg["nemo"]["wt"])), "builds": {}}
    for b in encs:
        summary["builds"][b] = {
            "wer_truth": float(np.mean(agg[b]["wt"])),
            "wer_truth_median": float(np.median(agg[b]["wt"])),
            "drift_nemo": float(np.mean(agg[b]["vn"])),
            "exact_match": agg[b]["exact"] / n if n else float("nan"),
        }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=2))

    print(f"\n=== SUMMARY  LibriSpeech test-clean  n={n} ===")
    print(f"  nemo            WER_truth={summary['nemo_wer_truth']:.4f}")
    print(f"  {'build':14} {'WER_truth':>10} {'WER_med':>8} {'drift_NeMo':>11} {'exact%':>7}")
    for b in encs:
        d = summary["builds"][b]
        print(f"  {b:14} {d['wer_truth']:>10.4f} {d['wer_truth_median']:>8.4f} {d['drift_nemo']:>11.4f} {d['exact_match']:>6.0%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
