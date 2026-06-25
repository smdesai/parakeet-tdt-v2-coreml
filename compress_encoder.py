#!/usr/bin/env python3
"""Post-training compression of the validated FP16 encoder.mlpackage.

The FP16 encoder is 1.1GB. The question: can we shrink it without the fidelity
loss the shipped 6-bit palettized baseline shows (27.6x more NeMo-drift than FP16)?

We apply coremltools.optimize.coreml weight compression DIRECTLY to the FP16
.mlpackage (no NeMo round-trip, no re-conversion). Each scheme is a separate
build dir so the existing diag_* harness can gate it on rel_l2 + WER.

Schemes (bytes/weight, est size from 1.1GB FP16 base):
  int8         INT8 linear, per-channel symmetric   ~1.0  ~0.56GB  (gentlest)
  palett8      8-bit palettize, kmeans, per-grouped  ~1.0  ~0.56GB
  palett6      6-bit palettize, kmeans               ~0.75 ~0.42GB  (~ shipped)
  palett4      4-bit palettize, kmeans               ~0.5  ~0.28GB  (aggressive)

Usage:
  compress_encoder.py --scheme int8
  compress_encoder.py --scheme palett8
  compress_encoder.py --all
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

FP16_DIR = Path("./parakeet_coreml_v2_fp16")
SRC_ENC = FP16_DIR / "parakeet_encoder.mlpackage"
OUT_ROOT = Path(".")


def _dir_size_gb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def make_config(scheme: str) -> cto.OptimizationConfig:
    if scheme == "int8":
        op = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel",
            weight_threshold=2048,
        )
    elif scheme == "palett8":
        op = cto.OpPalettizerConfig(mode="kmeans", nbits=8, granularity="per_tensor",
                                    weight_threshold=2048)
    elif scheme == "palett6":
        op = cto.OpPalettizerConfig(mode="kmeans", nbits=6, granularity="per_tensor",
                                    weight_threshold=2048)
    elif scheme == "palett4":
        op = cto.OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_tensor",
                                    weight_threshold=2048)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return cto.OptimizationConfig(global_config=op)


def compress(scheme: str) -> Path:
    out_dir = OUT_ROOT / f"parakeet_coreml_v2_{scheme}"
    out_dir.mkdir(exist_ok=True)
    out_enc = out_dir / "parakeet_encoder.mlpackage"
    if out_enc.exists():
        shutil.rmtree(out_enc)

    print(f"\n=== scheme={scheme} ===")
    print(f"loading FP16 encoder {SRC_ENC} ({_dir_size_gb(SRC_ENC):.2f}GB)…")
    mdl = ct.models.MLModel(str(SRC_ENC))

    is_lin = scheme == "int8"
    fn = cto.linear_quantize_weights if is_lin else cto.palettize_weights
    cfg = make_config(scheme)
    t0 = time.time()
    comp = fn(mdl, cfg)
    comp.save(str(out_enc))
    dt = time.time() - t0
    sz = _dir_size_gb(out_enc)
    print(f"  -> {out_enc}  {sz:.2f}GB  ({dt:.0f}s)")

    # copy the FP16 preprocessor alongside so the dir is a usable pair
    src_prep = FP16_DIR / "parakeet_preprocessor.mlpackage"
    dst_prep = out_dir / "parakeet_preprocessor.mlpackage"
    if src_prep.exists() and not dst_prep.exists():
        shutil.copytree(src_prep, dst_prep)
    return out_enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["int8", "palett8", "palett6", "palett4"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not SRC_ENC.exists():
        raise SystemExit(f"missing source FP16 encoder: {SRC_ENC}")

    schemes = ["int8", "palett8", "palett6", "palett4"] if args.all else [args.scheme]
    if schemes == [None]:
        raise SystemExit("pass --scheme <name> or --all")

    results = []
    base = _dir_size_gb(SRC_ENC)
    for s in schemes:
        p = compress(s)
        results.append((s, _dir_size_gb(p)))

    print(f"\n=== sizes (FP16 base {base:.2f}GB) ===")
    print(f"  {'scheme':10} {'size_GB':>8} {'vs_fp16':>8}")
    for s, sz in results:
        print(f"  {s:10} {sz:>8.2f} {sz/base:>7.0%}")


if __name__ == "__main__":
    main()
