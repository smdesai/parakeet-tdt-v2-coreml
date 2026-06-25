#!/usr/bin/env python3
"""Quantify the decode-restart seam penalty on long (>15s) clips, and whether
carrying RNN-T decode state across the fixed 15s windows removes it.

The INT8 v2 encoder is fixed at 240000 samples (15s). For long audio the app
must run it over multiple windows. Two ways to turn N window-encodings into text:

  A (independent / current v2 harness):
      decode each window's encoder output from a BLANK decoder state, then
      string-join the per-window texts. Every 15s seam restarts the predictor.

  B (carried-state / parakeet-unified batch-path trick):
      concatenate the per-window encoder outputs along time into one
      [1,1024,ΣT] stream and run a SINGLE RNN-T decode. The decoder sees one
      continuous token stream -> no seam restart. No overlap, no stitcher.

Ceiling = NeMo full single-pass (encoder sees the whole clip at once).

  A->B  isolates the decode-restart seam penalty.
  B->NeMo isolates the residual encoder-context-truncation penalty.

All three share identical text normalization and the SAME INT8 encoder windows
for A and B, so the only difference between A and B is decode continuity.

Usage:
  exp_carried_state.py                       # lsc (all 9), INT8 final
  exp_carried_state.py --build shipped_6bit_orig
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
    CoreMLEncoder, SR, WIN, decode_encoder, load_16k_mono, norm, windows_of,
)

NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
# Deployable encoder in .mlpackage form (INT8 encoder + FP16 preprocessor) — the
# pair emitted by compress_encoder.py. The compiled .mlmodelc runtime dir
# (parakeet_coreml_v2_final) carries no .mlpackage, so the gate loads from here.
FINAL_DIR = Path("./parakeet_coreml_v2_int8")
ORIG_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-orig")
LSC_ROOT = "/Users/sdesai/Tools/AI/lsc-tests2"

KNOWN = {
    "final": (FINAL_DIR / "parakeet_preprocessor.mlpackage", FINAL_DIR / "parakeet_encoder.mlpackage"),
    "shipped_6bit_orig": (ORIG_DIR / "Preprocessor.mlmodelc", ORIG_DIR / "Encoder.mlmodelc"),
}


def encode_windows(enc: CoreMLEncoder, wav: np.ndarray):
    """Return list of (enc_out [1,1024,T], enc_len int) per 15s window, each
    trimmed to its valid frame count."""
    outs = []
    for seg, true_len in windows_of(wav):
        e, el = enc.encode(seg, true_len)
        T = int(np.asarray(el).reshape(-1)[0])
        T = min(T, e.shape[2])
        outs.append((e[:, :, :T], T))
    return outs


def decode_independent(m, win_outs) -> str:
    """Strategy A: decode each window from blank; join texts."""
    parts = []
    for e, T in win_outs:
        txt = decode_encoder(m, e, np.array([T], dtype=np.int32))
        if txt.strip():
            parts.append(txt.strip())
    return " ".join(parts)


def decode_carried(m, win_outs) -> str:
    """Strategy B: concat encoder outputs along time -> single decode."""
    if not win_outs:
        return ""
    cat = np.concatenate([e for e, _ in win_outs], axis=2)  # [1,1024,ΣT]
    total = sum(T for _, T in win_outs)
    return decode_encoder(m, cat, np.array([total], dtype=np.int32))


# Strategy C: OVERLAPPING windows (parakeet-unified streaming trick). Slide a 15s
# window with left+right context, but emit only the center frames, concatenate the
# emitted frame slices, single decode. This gives every emitted frame real
# cross-boundary encoder context — the thing B lacks.
CTX_S = 2.5            # context each side (s)
CENTER_S = WIN / SR - 2 * CTX_S   # 10s center


def encode_overlap_emit(enc: CoreMLEncoder, wav: np.ndarray):
    """Yield center-frame encoder slices from overlapping 15s windows. Avoids
    left zero-padding (which would break tail-only length masking) by clamping the
    window start to 0 and shifting the emit region instead."""
    n = wav.shape[0]
    ctx = int(CTX_S * SR)
    center = int(CENTER_S * SR)
    outs = []
    c0 = 0
    while c0 < n:
        c1 = min(c0 + center, n)                 # center region [c0, c1) in samples
        win_start = max(0, c0 - ctx)             # no left pad: clamp to 0
        win_end = win_start + WIN
        seg = wav[win_start:win_end]
        true_len = min(seg.shape[0], WIN)
        if seg.shape[0] < WIN:                    # right pad only (tail mask ok)
            seg = np.pad(seg, (0, WIN - seg.shape[0]))
        e, el = enc.encode(seg, true_len)
        T = min(int(np.asarray(el).reshape(-1)[0]), e.shape[2])
        # map the center sample-region [c0,c1) to encoder frames within this window
        ratio = T / max(1, true_len)
        fs = int(round((c0 - win_start) * ratio))
        fe = int(round((c1 - win_start) * ratio))
        fs = max(0, min(fs, T))
        fe = max(fs, min(fe, T))
        if fe > fs:
            outs.append((e[:, :, fs:fe], fe - fs))
        c0 += center
    return outs


def decode_overlap(m, enc, wav) -> str:
    """Strategy C: overlap windows + SINGLE continuous decode (carried state)."""
    win_outs = encode_overlap_emit(enc, wav)
    if not win_outs:
        return ""
    cat = np.concatenate([e for e, _ in win_outs], axis=2)
    total = sum(T for _, T in win_outs)
    return decode_encoder(m, cat, np.array([total], dtype=np.int32))


def decode_overlap_indep(m, enc, wav) -> str:
    """Strategy D: overlap windows but decode each emitted center from BLANK and
    join. Isolates the overlap contribution from the carried-state contribution:
    D vs C tells us whether single-continuous-decode matters once overlap is in."""
    win_outs = encode_overlap_emit(enc, wav)
    parts = []
    for e, T in win_outs:
        txt = decode_encoder(m, e, np.array([T], dtype=np.int32))
        if txt.strip():
            parts.append(txt.strip())
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="final", choices=list(KNOWN))
    ap.add_argument("--out", default="./bench_results/v2_carried_state_lsc.json")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    prep, encp = KNOWN[args.build]
    enc = CoreMLEncoder(prep, encp)

    def nemo_singlepass(wav):
        with torch.no_grad():
            mel_t, mel_len_t = m.preprocessor(
                input_signal=torch.tensor(wav).unsqueeze(0).float(),
                length=torch.tensor([wav.shape[0]]),
            )
            enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
        return decode_encoder(m, enc_t.cpu().numpy(), enc_len_t.cpu().numpy())

    wavs = sorted(glob.glob(os.path.join(LSC_ROOT, "*.wav")))
    print(f"lsc clips: {len(wavs)}  (build={args.build})\n")

    rows = []
    agg = {k: [] for k in ("A_indep", "B_carried", "C_overlap", "D_overlap_indep", "nemo")}
    for wpath in wavs:
        base = wpath[:-4]
        gt_file = next((c for c in (base + ".text.txt", base + ".txt") if os.path.exists(c)), None)
        if not gt_file:
            continue
        gt = norm(Path(gt_file).read_text())
        wav = load_16k_mono(wpath)
        nwin = max(1, (wav.shape[0] + WIN - 1) // WIN)

        win_outs = encode_windows(enc, wav)
        txt_A = decode_independent(m, win_outs)
        txt_B = decode_carried(m, win_outs)
        txt_C = decode_overlap(m, enc, wav)
        txt_D = decode_overlap_indep(m, enc, wav)
        txt_N = nemo_singlepass(wav)

        wer_A = jiwer.wer(gt, norm(txt_A))
        wer_B = jiwer.wer(gt, norm(txt_B))
        wer_C = jiwer.wer(gt, norm(txt_C))
        wer_D = jiwer.wer(gt, norm(txt_D))
        wer_N = jiwer.wer(gt, norm(txt_N))
        agg["A_indep"].append(wer_A)
        agg["B_carried"].append(wer_B)
        agg["C_overlap"].append(wer_C)
        agg["D_overlap_indep"].append(wer_D)
        agg["nemo"].append(wer_N)

        rows.append({
            "clip": os.path.basename(wpath), "dur_s": round(wav.shape[0] / SR, 1),
            "windows": nwin, "gt_words": len(gt.split()),
            "A_indep_wer": round(wer_A, 4), "B_carried_wer": round(wer_B, 4),
            "C_overlap_wer": round(wer_C, 4), "D_overlap_indep_wer": round(wer_D, 4),
            "nemo_wer": round(wer_N, 4),
            "AB_delta": round(wer_A - wer_B, 4), "AC_delta": round(wer_A - wer_C, 4),
            "CD_delta": round(wer_C - wer_D, 4),
            "_A": txt_A, "_B": txt_B, "_C": txt_C, "_D": txt_D, "_N": txt_N,
        })
        print(f"  {os.path.basename(wpath):12} {round(wav.shape[0]/SR,1):5}s {nwin}w  "
              f"A={wer_A:.4f}  B={wer_B:.4f}  C={wer_C:.4f}  D={wer_D:.4f}  NeMo={wer_N:.4f}")

    summary = {"build": args.build, "n": len(rows),
               "A_indep_wer": float(np.mean(agg["A_indep"])),
               "B_carried_wer": float(np.mean(agg["B_carried"])),
               "C_overlap_wer": float(np.mean(agg["C_overlap"])),
               "D_overlap_indep_wer": float(np.mean(agg["D_overlap_indep"])),
               "nemo_wer": float(np.mean(agg["nemo"])),
               "seam_penalty_A_minus_B": float(np.mean(agg["A_indep"]) - np.mean(agg["B_carried"])),
               "overlap_gain_A_minus_C": float(np.mean(agg["A_indep"]) - np.mean(agg["C_overlap"])),
               "carried_gain_D_minus_C": float(np.mean(agg["D_overlap_indep"]) - np.mean(agg["C_overlap"])),
               "context_penalty_C_minus_nemo": float(np.mean(agg["C_overlap"]) - np.mean(agg["nemo"]))}

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_clip": rows}, indent=2))

    print(f"\n=== SUMMARY  lsc (>15s clips)  build={args.build}  n={summary['n']} ===")
    print(f"  A independent (current)       WER = {summary['A_indep_wer']:.4f}")
    print(f"  B carried-state (no overlap)  WER = {summary['B_carried_wer']:.4f}")
    print(f"  C overlap + carried decode    WER = {summary['C_overlap_wer']:.4f}")
    print(f"  D overlap + independent decode WER = {summary['D_overlap_indep_wer']:.4f}")
    print(f"  NeMo single-pass              WER = {summary['nemo_wer']:.4f}")
    print(f"  -> seam penalty (A-B)             = {summary['seam_penalty_A_minus_B']:+.4f}")
    print(f"  -> overlap gain (A-C)             = {summary['overlap_gain_A_minus_C']:+.4f}")
    print(f"  -> carried-vs-indep w/ overlap (D-C) = {summary['carried_gain_D_minus_C']:+.4f}")
    print(f"  -> context penalty (C-NeMo)       = {summary['context_penalty_C_minus_nemo']:+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
