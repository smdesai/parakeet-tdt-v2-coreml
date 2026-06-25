#!/usr/bin/env python3
"""Long-form (>15s) transcription for the fixed-15s Parakeet-TDT-v2 CoreML encoder.

Implements the recommendation from the long-clip decode-strategy ablation
(`exp_carried_state.py`, commits 318e245 + c2704af): **overlapping windows +
ONE continuous RNN-T decode**. On lsc (9 clips, 31–66s) this scored WER 0.0726
vs NeMo full single-pass 0.0667 — recovering ~78% of the long-clip gap — while
the two single-ingredient variants both fail (carried-state-no-overlap 0.0932;
overlap-but-per-window-decode 0.2715).

Why both ingredients are required:
  * Overlap: each emitted encoder frame is computed inside a window that carries
    real acoustic context (`ctx_samples`) on both sides, so frames near a seam
    are no longer truncated. This closes the encoder-context gap.
  * Single continuous decode: the emitted center-frame slices are concatenated
    into one [1,1024,ΣT] stream and decoded in a SINGLE RNN-T pass. Decoding the
    slices independently restarts the predictor cold mid-utterance and silently
    drops the tokens straddling each cut (-20-30% words). One decode avoids that.

The encoder is fixed at WIN=240000 samples (15s). We slide a window whose center
(`center_samples`) tiles the audio exactly; each window also reads `ctx_samples`
of look-around that is encoded but NOT emitted. The first window has no left
context (start of audio) and the last has no right context (end of audio) — same
as NeMo sees at the true boundaries.

Public API:
  transcribe_longform(encoder_fn, nemo_model, wav, cfg) -> str   # Strategy C
  transcribe_baseline(encoder_fn, nemo_model, wav)      -> str   # Strategy A (current)
  plan_windows(n_samples, cfg) -> list[WindowSpec]              # pure, testable

`encoder_fn(window_f32[WIN], true_len) -> (enc[1,1024,T], enc_len[1])` — satisfied
by both `CoreMLEncoder.encode` and the NeMo encoder adapter below.

CLI:
  longform_transcribe.py --audio clip.wav [--build final]            # transcribe one file
  longform_transcribe.py --dataset lsc [--build final]               # A vs C vs NeMo WER
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from diag_dataset_wer import (
    CoreMLEncoder, SR, WIN, decode_encoder, load_16k_mono, norm, windows_of,
)

NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
# Deployable encoder in .mlpackage form (INT8 encoder + FP16 preprocessor) — the
# pair emitted by compress_encoder.py. The compiled .mlmodelc runtime dir
# (parakeet_coreml_v2_final) carries no .mlpackage, so this loads from here.
FINAL_DIR = Path("./parakeet_coreml_v2_int8")
ORIG_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-orig")
LSC_ROOT = "/Users/sdesai/Tools/AI/lsc-tests2"

KNOWN = {
    "final": (FINAL_DIR / "parakeet_preprocessor.mlpackage", FINAL_DIR / "parakeet_encoder.mlpackage"),
    "shipped_6bit_orig": (ORIG_DIR / "Preprocessor.mlmodelc", ORIG_DIR / "Encoder.mlmodelc"),
}


# ---------------------------------------------------------------------------
# Windowing (pure — no model, unit-testable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LongformConfig:
    """Sliding-window geometry. center_samples = win_samples - 2*ctx_samples.

    Defaults: 15s window, 2.5s context each side -> 10s emitted center. 2.5s was
    the value validated in the ablation; it trades encoder calls (1.5x at 10s
    center) for boundary context. Larger ctx = more context but more passes.
    """
    win_samples: int = WIN          # fixed encoder input (240000 = 15s)
    ctx_samples: int = 40000        # 2.5s @ 16kHz, each side
    sr: int = SR

    @property
    def center_samples(self) -> int:
        c = self.win_samples - 2 * self.ctx_samples
        if c <= 0:
            raise ValueError(f"ctx_samples={self.ctx_samples} too large for win {self.win_samples}")
        return c


@dataclass(frozen=True)
class WindowSpec:
    """One encoder window. The window feeds samples [win_start, win_start+true_len)
    (right-zero-padded to win_samples); only the center [center_lo, center_hi) is
    emitted into the final decode stream."""
    win_start: int      # first sample of the encoder window
    true_len: int       # real (unpadded) samples in the window
    center_lo: int      # first sample of the emitted region (absolute)
    center_hi: int      # one-past-last emitted sample (absolute)


def plan_windows(n_samples: int, cfg: LongformConfig) -> list[WindowSpec]:
    """Tile [0, n_samples) into overlapping encoder windows whose emitted centers
    are contiguous and non-overlapping (they exactly cover the audio).

    Short audio (<= one window) collapses to a single full-context window with no
    overlap, matching the normal fast path and avoiding any seam."""
    if n_samples <= cfg.win_samples:
        return [WindowSpec(win_start=0, true_len=n_samples, center_lo=0, center_hi=n_samples)]

    specs: list[WindowSpec] = []
    center = cfg.center_samples
    c0 = 0
    while c0 < n_samples:
        c1 = min(c0 + center, n_samples)
        win_start = max(0, c0 - cfg.ctx_samples)            # no left pad: clamp, emit shifts
        true_len = min(cfg.win_samples, n_samples - win_start)
        specs.append(WindowSpec(win_start=win_start, true_len=true_len,
                                center_lo=c0, center_hi=c1))
        c0 += center
    return specs


# ---------------------------------------------------------------------------
# Encoding + emit-slice extraction
# ---------------------------------------------------------------------------
def _slice_for_window(enc: np.ndarray, t_valid: int, spec: WindowSpec) -> np.ndarray:
    """Map the window's emitted sample region to encoder frames and return that
    slice [1,1024,fe-fs]. Uses the window's own samples->frames ratio so it is
    robust to the encoder's exact subsampling factor."""
    ratio = t_valid / max(1, spec.true_len)
    fs = int(round((spec.center_lo - spec.win_start) * ratio))
    fe = int(round((spec.center_hi - spec.win_start) * ratio))
    fs = max(0, min(fs, t_valid))
    fe = max(fs, min(fe, t_valid))
    return enc[:, :, fs:fe]


def emit_slices(encoder_fn, wav: np.ndarray, cfg: LongformConfig) -> list[np.ndarray]:
    """Encode each planned window and return the list of emitted center-frame
    slices, in order. encoder_fn(window_f32[WIN], true_len) -> (enc, enc_len)."""
    slices: list[np.ndarray] = []
    for spec in plan_windows(wav.shape[0], cfg):
        seg = wav[spec.win_start:spec.win_start + cfg.win_samples]
        if seg.shape[0] < cfg.win_samples:                  # right pad only (tail mask)
            seg = np.pad(seg, (0, cfg.win_samples - seg.shape[0]))
        enc, enc_len = encoder_fn(seg.astype(np.float32), spec.true_len)
        enc = np.asarray(enc)
        t_valid = min(int(np.asarray(enc_len).reshape(-1)[0]), enc.shape[2])
        sl = _slice_for_window(enc, t_valid, spec)
        if sl.shape[2] > 0:
            slices.append(sl)
    return slices


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def transcribe_longform(encoder_fn, nemo_model, wav: np.ndarray,
                        cfg: LongformConfig | None = None) -> str:
    """Strategy C (recommended): overlapping windows + ONE continuous decode."""
    cfg = cfg or LongformConfig()
    slices = emit_slices(encoder_fn, wav, cfg)
    if not slices:
        return ""
    cat = np.concatenate(slices, axis=2)                    # [1,1024,ΣT]
    total = cat.shape[2]
    return decode_encoder(nemo_model, cat, np.array([total], dtype=np.int32))


def transcribe_baseline(encoder_fn, nemo_model, wav: np.ndarray) -> str:
    """Strategy A (current harness): non-overlapping 15s windows, decode each from
    blank, string-join. Kept for direct comparison."""
    parts = []
    for seg, true_len in windows_of(wav):
        enc, enc_len = encoder_fn(seg, true_len)
        txt = decode_encoder(nemo_model, enc, enc_len)
        if txt.strip():
            parts.append(txt.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def nemo_singlepass(model, wav: np.ndarray) -> str:
    """True NeMo ceiling: encode the WHOLE clip in one pass (no windowing), then
    decode. NeMo's encoder accepts arbitrary length, unlike the fixed CoreML one."""
    with torch.no_grad():
        mel_t, mel_len_t = model.preprocessor(
            input_signal=torch.tensor(wav).unsqueeze(0).float(),
            length=torch.tensor([wav.shape[0]]),
        )
        enc_t, enc_len_t = model.encoder(audio_signal=mel_t, length=mel_len_t)
    return decode_encoder(model, enc_t.cpu().numpy(), enc_len_t.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="transcribe a single wav with Strategy C")
    ap.add_argument("--dataset", help="run A/C/NeMo WER over a dir of wavs (e.g. lsc path or 'lsc')")
    ap.add_argument("--build", default="final", choices=list(KNOWN))
    ap.add_argument("--ctx-s", type=float, default=2.5, help="context seconds each side")
    ap.add_argument("--out", default="./bench_results/v2_longform_lsc.json")
    args = ap.parse_args()

    cfg = LongformConfig(ctx_samples=int(args.ctx_s * SR))

    import nemo.collections.asr as nemo_asr
    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    prep, encp = KNOWN[args.build]
    enc = CoreMLEncoder(prep, encp)
    enc_fn = lambda window, true_len: enc.encode(window, true_len)

    if args.audio:
        wav = load_16k_mono(args.audio)
        n_win = len(plan_windows(wav.shape[0], cfg))
        print(f"{os.path.basename(args.audio)}  {wav.shape[0]/SR:.1f}s  "
              f"ctx={args.ctx_s}s  center={cfg.center_samples/SR:.1f}s  windows={n_win}\n")
        print(transcribe_longform(enc_fn, m, wav, cfg))
        return

    root = LSC_ROOT if (not args.dataset or args.dataset == "lsc") else args.dataset
    import jiwer
    wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
    print(f"dataset={root}  clips={len(wavs)}  build={args.build}  ctx={args.ctx_s}s\n")

    rows, agg = [], {k: [] for k in ("A", "C", "nemo")}
    for wpath in wavs:
        base = wpath[:-4]
        gt_file = next((c for c in (base + ".text.txt", base + ".txt") if os.path.exists(c)), None)
        if not gt_file:
            continue
        gt = norm(Path(gt_file).read_text())
        wav = load_16k_mono(wpath)
        n_win = len(plan_windows(wav.shape[0], cfg))

        txt_A = transcribe_baseline(enc_fn, m, wav)
        txt_C = transcribe_longform(enc_fn, m, wav, cfg)
        txt_N = nemo_singlepass(m, wav)

        wer_A = jiwer.wer(gt, norm(txt_A))
        wer_C = jiwer.wer(gt, norm(txt_C))
        wer_N = jiwer.wer(gt, norm(txt_N))
        for k, v in (("A", wer_A), ("C", wer_C), ("nemo", wer_N)):
            agg[k].append(v)
        rows.append({"clip": os.path.basename(wpath), "dur_s": round(wav.shape[0] / SR, 1),
                     "windows_C": n_win, "A_wer": round(wer_A, 4),
                     "C_wer": round(wer_C, 4), "nemo_wer": round(wer_N, 4),
                     "_A": txt_A, "_C": txt_C, "_N": txt_N})
        print(f"  {os.path.basename(wpath):12} {wav.shape[0]/SR:5.1f}s {n_win}w  "
              f"A={wer_A:.4f}  C={wer_C:.4f}  NeMo={wer_N:.4f}")

    summary = {"build": args.build, "ctx_s": args.ctx_s, "n": len(rows),
               "A_wer": float(np.mean(agg["A"])), "C_wer": float(np.mean(agg["C"])),
               "nemo_wer": float(np.mean(agg["nemo"])),
               "C_gain_over_A": float(np.mean(agg["A"]) - np.mean(agg["C"])),
               "C_gap_to_nemo": float(np.mean(agg["C"]) - np.mean(agg["nemo"]))}
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_clip": rows}, indent=2))

    print(f"\n=== SUMMARY  build={args.build}  ctx={args.ctx_s}s  n={summary['n']} ===")
    print(f"  A baseline (current)  WER = {summary['A_wer']:.4f}")
    print(f"  C longform (this)     WER = {summary['C_wer']:.4f}")
    print(f"  NeMo                  WER = {summary['nemo_wer']:.4f}")
    print(f"  -> C gain over A          = {summary['C_gain_over_A']:+.4f}")
    print(f"  -> C residual gap to NeMo = {summary['C_gap_to_nemo']:+.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
