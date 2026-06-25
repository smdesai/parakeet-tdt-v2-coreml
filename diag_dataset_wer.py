#!/usr/bin/env python3
"""Broader-dataset WER: FP16 CoreML vs shipped 6-bit baseline vs NeMo, on hard audio.

Methodology (encoder-isolated, end-to-end WER vs ground truth):
  For each clip: resample->16k mono, split into <=15s windows (240000 samples,
  zero-padded with true length passed so the model masks correctly). For each
  system (nemo / fp16 / shipped_6bit) run that encoder on every window, decode
  with NeMo's RNNT decoder (swap-in), concatenate window texts, and score WER
  against the dataset's ground-truth .text.txt.

All three systems share IDENTICAL chunking + NeMo decode, so WER differences are
purely encoder-driven. NeMo here is the upper bound (its own encoder, chunked the
same way). Models are loaded once and reused across all clips.

Usage:
  diag_dataset_wer.py --dataset fda   [--limit N] [--offset K]
  diag_dataset_wer.py --dataset lsc
  diag_dataset_wer.py --dataset both  --systems fp16,shipped_6bit,nemo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import coremltools as ct
import jiwer
import numpy as np
import soundfile as sf
import torch

NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
FP32_DIR = Path("./parakeet_coreml_v2_fp32")
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")
CU = ct.ComputeUnit.CPU_ONLY
SR = 16000
WIN = 240000  # 15s @ 16kHz

DATASETS = {
    "fda": "/Users/sdesai/Tools/AI/test-dataset-fda-extended/test_cases",
    "lsc": "/Users/sdesai/Tools/AI/lsc-tests2",
    "earnings22": "/Users/sdesai/Library/Application Support/FluidAudio/earnings22-kws/test-dataset",
}


# ---------- text normalization for WER ----------
_punct = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    s = s.lower()
    s = _punct.sub(" ", s)
    return " ".join(s.split())


# ---------- model I/O ----------
def io_names(path: Path):
    if str(path).endswith(".mlmodelc"):
        meta = json.loads((path / "metadata.json").read_text())
        m = meta[0] if isinstance(meta, list) else meta
        return ([i["name"] for i in m.get("inputSchema", [])],
                [o["name"] for o in m.get("outputSchema", [])])
    spec = ct.models.MLModel(str(path), compute_units=CU).get_spec()
    return ([i.name for i in spec.description.input],
            [o.name for o in spec.description.output])


def load_model(path: Path):
    p = str(path)
    if p.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(p, compute_units=CU)
    return ct.models.MLModel(p, compute_units=CU)


class CoreMLEncoder:
    """Cached preprocessor+encoder pair; encodes a single 15s window."""

    def __init__(self, prep_path: Path, enc_path: Path):
        self.p_in, self.p_out = io_names(prep_path)
        self.e_in, self.e_out = io_names(enc_path)
        self.prep = load_model(prep_path)
        self.enc = load_model(enc_path)

    def encode(self, window: np.ndarray, true_len: int | None = None):
        """window: float32 [WIN] padded; true_len: real (unpadded) sample count.
        Returns (enc [1,1024,T], enc_len [1])."""
        if true_len is None:
            true_len = window.shape[0]
        prep_inputs = {self.p_in[0]: window.astype(np.float32).reshape(1, -1)}
        if "audio_length" in self.p_in:
            prep_inputs["audio_length"] = np.array([true_len], dtype=np.int32)
        po = self.prep.predict(prep_inputs)
        mel = po[self.p_out[0]]
        mel_len = po["mel_length"] if "mel_length" in po else np.array([mel.shape[2]], dtype=np.int32)
        enc_inputs = {self.e_in[0]: mel.astype(np.float32)}
        if "mel_length" in self.e_in:
            enc_inputs["mel_length"] = np.asarray(mel_len, dtype=np.int32)
        eo = self.enc.predict(enc_inputs)
        enc = eo[self.e_out[0]]
        enc_len = eo.get("encoder_length", np.array([enc.shape[2]], dtype=np.int32))
        return enc, np.asarray(enc_len)


def decode_encoder(model, enc: np.ndarray, enc_len: np.ndarray) -> str:
    enc_t = torch.from_numpy(np.asarray(enc)).float()
    len_t = torch.from_numpy(np.asarray(enc_len)).long().reshape(-1)
    with torch.no_grad():
        preds = model.decoding.rnnt_decoder_predictions_tensor(enc_t, len_t, return_hypotheses=True)
    h = preds[0] if isinstance(preds, tuple) else preds
    if isinstance(h, list):
        h = h[0]
    return h.text if hasattr(h, "text") else str(h)


# ---------- audio ----------
def load_16k_mono(path: str) -> np.ndarray:
    import torchaudio
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SR:
        t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(t, sr, SR).squeeze(0).numpy()
    return wav.astype(np.float32)


def windows_of(wav: np.ndarray):
    """Yield zero-padded 15s windows with their true length."""
    n = wav.shape[0]
    nwin = max(1, (n + WIN - 1) // WIN)
    for w in range(nwin):
        seg = wav[w * WIN:(w + 1) * WIN]
        true_len = seg.shape[0]
        if true_len < WIN:
            seg = np.pad(seg, (0, WIN - true_len))
        yield seg, true_len


# ---------- per-clip transcription ----------
def transcribe_chunked(encoder_fn, nemo_model, wav: np.ndarray) -> str:
    """encoder_fn(window, true_len) -> (enc, enc_len). Concatenate decoded windows."""
    parts = []
    for seg, true_len in windows_of(wav):
        enc, enc_len = encoder_fn(seg, true_len)
        txt = decode_encoder(nemo_model, enc, enc_len)
        if txt.strip():
            parts.append(txt.strip())
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fda", "lsc", "earnings22", "both", "all"], default="both")
    ap.add_argument("--systems", default="nemo,fp16,shipped_6bit")
    ap.add_argument("--limit", type=int, default=0, help="max clips per dataset (0=all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="./bench_results/v2_dataset_wer.json")
    args = ap.parse_args()

    systems = args.systems.split(",")
    import nemo.collections.asr as nemo_asr
    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    # cache encoders
    enc_cache = {}
    if "fp16" in systems:
        enc_cache["fp16"] = CoreMLEncoder(FP16_DIR / "parakeet_preprocessor.mlpackage",
                                          FP16_DIR / "parakeet_encoder.mlpackage")
    if "fp32" in systems:
        enc_cache["fp32"] = CoreMLEncoder(FP32_DIR / "parakeet_preprocessor.mlpackage",
                                          FP32_DIR / "parakeet_encoder.mlpackage")
    if "shipped_6bit" in systems:
        enc_cache["shipped_6bit"] = CoreMLEncoder(BASE_DIR / "Preprocessor.mlmodelc",
                                                  BASE_DIR / "Encoder.mlmodelc")

    def nemo_encoder_fn(window, true_len):
        with torch.no_grad():
            mel_t, mel_len_t = m.preprocessor(
                input_signal=torch.tensor(window).unsqueeze(0).float(),
                length=torch.tensor([true_len]),
            )
            enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
        return enc_t.cpu().numpy(), enc_len_t.cpu().numpy()

    def sys_fn(name):
        if name == "nemo":
            return nemo_encoder_fn
        e = enc_cache[name]
        return lambda window, true_len: e.encode(window, true_len)

    if args.dataset == "both":
        ds_list = ["fda", "lsc"]
    elif args.dataset == "all":
        ds_list = ["fda", "lsc", "earnings22"]
    else:
        ds_list = [args.dataset]
    per_clip = []
    agg = {ds: {s: {"wer_truth": [], "wer_vs_nemo": []} for s in systems} for ds in ds_list}

    for ds in ds_list:
        root = DATASETS[ds]
        wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
        if args.offset:
            wavs = wavs[args.offset:]
        if args.limit:
            wavs = wavs[:args.limit]
        print(f"\n=== dataset={ds}  clips={len(wavs)} ===")
        for i, wpath in enumerate(wavs):
            tpath = wpath[:-4]
            # find ground truth text
            gt_file = None
            for cand in (tpath + ".text.txt", tpath + ".txt"):
                if os.path.exists(cand):
                    gt_file = cand
                    break
            if not gt_file:
                continue
            gt = norm(Path(gt_file).read_text())
            wav = load_16k_mono(wpath)

            texts = {}
            for s in systems:
                texts[s] = transcribe_chunked(sys_fn(s), m, wav)

            row = {"dataset": ds, "clip": os.path.basename(wpath),
                   "dur_s": round(wav.shape[0] / SR, 1), "gt_words": len(gt.split())}
            nemo_txt = texts.get("nemo")
            for s in systems:
                wer_t = jiwer.wer(gt, norm(texts[s])) if gt.strip() else float("nan")
                row[f"{s}_wer_truth"] = wer_t
                agg[ds][s]["wer_truth"].append(wer_t)
                if nemo_txt is not None and s != "nemo":
                    wer_n = jiwer.wer(norm(nemo_txt), norm(texts[s]))
                    row[f"{s}_wer_vs_nemo"] = wer_n
                    agg[ds][s]["wer_vs_nemo"].append(wer_n)
            row["_texts"] = {s: texts[s] for s in systems}
            per_clip.append(row)

            short = "  ".join(f"{s}={row[f'{s}_wer_truth']:.3f}" for s in systems)
            print(f"  [{ds} {i+1}/{len(wavs)}] {row['clip']} {row['dur_s']}s  WERvtruth: {short}")

    # aggregate
    summary = {}
    for ds in ds_list:
        summary[ds] = {}
        for s in systems:
            wt = [x for x in agg[ds][s]["wer_truth"] if not np.isnan(x)]
            wn = [x for x in agg[ds][s]["wer_vs_nemo"] if not np.isnan(x)]
            summary[ds][s] = {
                "n": len(wt),
                "wer_truth_mean": float(np.mean(wt)) if wt else None,
                "wer_truth_median": float(np.median(wt)) if wt else None,
                "wer_vs_nemo_mean": float(np.mean(wn)) if wn else None,
            }

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"systems": systems, "summary": summary, "per_clip": per_clip}, indent=2))

    print("\n=== SUMMARY (WER vs ground truth; lower is better) ===")
    for ds in ds_list:
        print(f"\ndataset={ds}")
        print(f"  {'system':14} {'n':>4} {'WER_truth_mean':>15} {'WER_truth_med':>14} {'WER_vs_nemo':>12}")
        for s in systems:
            d = summary[ds][s]
            vn = f"{d['wer_vs_nemo_mean']:.4f}" if d['wer_vs_nemo_mean'] is not None else "  -  "
            print(f"  {s:14} {d['n']:>4} {d['wer_truth_mean']:>15.4f} {d['wer_truth_median']:>14.4f} {vn:>12}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
