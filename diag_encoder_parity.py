#!/usr/bin/env python3
"""Focused diagnostic: why does encoder rel_l2 read ~0.68 when decode is identical?

Tests, on the same 15s clip:
  1. NeMo mel vs CoreML FP32 mel  (sanity; expected ~0)
  2. NeMo encoder(NeMo mel)  vs  CoreML encoder(CoreML mel)   [end-to-end]
  3. NeMo encoder(NeMo mel)  vs  CoreML encoder(NeMo mel)     [isolate encoder]
  4. transpose / length-trim checks to explain any large diff
Also loads the shipped 6-bit baseline via CompiledMLModel and reports its encoder parity.
"""
from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

AUDIO_15S = "./audio/yc_first_minute_16k_15s.wav"
NEMO_PATH = "/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo"
FP32_DIR = Path("./parakeet_coreml_v2_fp32")
FP16_DIR = Path("./parakeet_coreml_v2_fp16")
BASE_DIR = Path("/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2")
CU = ct.ComputeUnit.CPU_ONLY


def rel_l2(ref: np.ndarray, pred: np.ndarray) -> float:
    rn = np.linalg.norm(ref)
    return float(np.linalg.norm(pred - ref) / rn) if rn > 0 else float("nan")


def metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    if ref.shape != pred.shape:
        return {"shape_ref": list(ref.shape), "shape_pred": list(pred.shape), "note": "shape mismatch"}
    diff = pred - ref
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rel_l2": rel_l2(ref, pred),
        "ref_norm": float(np.linalg.norm(ref)),
        "pred_norm": float(np.linalg.norm(pred)),
    }


def load_model(path: Path):
    p = str(path)
    if p.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(p, compute_units=CU)
    return ct.models.MLModel(p, compute_units=CU)


def io_names(path: Path) -> tuple[list[str], list[str]]:
    """Return (inputs, outputs) feature names from metadata.json for .mlmodelc,
    or from the spec for .mlpackage."""
    if str(path).endswith(".mlmodelc"):
        meta = json.loads((path / "metadata.json").read_text())
        m = meta[0] if isinstance(meta, list) else meta
        return ([i["name"] for i in m.get("inputSchema", [])],
                [o["name"] for o in m.get("outputSchema", [])])
    spec = ct.models.MLModel(str(path), compute_units=CU).get_spec()
    return ([i.name for i in spec.description.input],
            [o.name for o in spec.description.output])


def coreml_mel_and_enc(prep_path: Path, enc_path: Path, audio: np.ndarray, feed_mel: np.ndarray | None = None):
    """Run CoreML preprocessor+encoder. If feed_mel given, skip prep and feed that mel to encoder."""
    p_in, p_out = io_names(prep_path)
    e_in, e_out = io_names(enc_path)
    prep = load_model(prep_path)
    enc = load_model(enc_path)

    if feed_mel is None:
        prep_inputs = {p_in[0]: audio.astype(np.float32).reshape(1, -1)}
        if "audio_length" in p_in:
            prep_inputs["audio_length"] = np.array([audio.shape[-1]], dtype=np.int32)
        po = prep.predict(prep_inputs)
        mel = po[p_out[0]]
        mel_len = po["mel_length"] if "mel_length" in po else np.array([mel.shape[2]], dtype=np.int32)
    else:
        mel = feed_mel.astype(np.float32)
        mel_len = np.array([mel.shape[2]], dtype=np.int32)

    enc_inputs = {e_in[0]: mel.astype(np.float32)}
    if "mel_length" in e_in:
        enc_inputs["mel_length"] = np.asarray(mel_len, dtype=np.int32)
    eo = enc.predict(enc_inputs)
    return mel, eo[e_out[0]], eo.get("encoder_length")


def main():
    import nemo.collections.asr as nemo_asr

    audio, sr = sf.read(AUDIO_15S)
    audio = audio.astype(np.float32)
    print(f"audio: {audio.shape} sr={sr}")

    print("loading nemo…")
    m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    m.eval()

    with torch.no_grad():
        mel_t, mel_len_t = m.preprocessor(
            input_signal=torch.tensor(audio).unsqueeze(0).float(),
            length=torch.tensor([len(audio)]),
        )
        enc_t, enc_len_t = m.encoder(audio_signal=mel_t, length=mel_len_t)
    nemo_mel = mel_t.cpu().numpy()
    nemo_enc = enc_t.cpu().numpy()
    nemo_enc_len = int(enc_len_t.cpu().numpy()[0])
    print(f"nemo mel {nemo_mel.shape}  enc {nemo_enc.shape}  enc_len={nemo_enc_len}")

    report = {"nemo": {"mel_shape": list(nemo_mel.shape), "enc_shape": list(nemo_enc.shape), "enc_len": nemo_enc_len}}

    for tag, d in [("fp32", FP32_DIR), ("fp16", FP16_DIR)]:
        prep = d / "parakeet_preprocessor.mlpackage"
        enc = d / "parakeet_encoder.mlpackage"
        if not (prep.exists() and enc.exists()):
            print(f"[{tag}] missing build, skip")
            continue
        # end-to-end (CoreML mel)
        cml_mel, cml_enc, cml_enc_len = coreml_mel_and_enc(prep, enc, audio)
        # isolate encoder (NeMo mel)
        _, cml_enc_nemomel, _ = coreml_mel_and_enc(prep, enc, audio, feed_mel=nemo_mel)

        entry = {
            "enc_shape": list(cml_enc.shape),
            "coreml_enc_len": (cml_enc_len.tolist() if cml_enc_len is not None else None),
            "mel_vs_nemo": metrics(nemo_mel, cml_mel),
            "enc_e2e_vs_nemo_full_T": metrics(nemo_enc, cml_enc),
            "enc_isolated_vs_nemo_full_T": metrics(nemo_enc, cml_enc_nemomel),
        }
        # length-trim: compare only the first enc_len valid frames
        if cml_enc.shape == nemo_enc.shape and cml_enc.ndim == 3:
            L = nemo_enc_len
            entry["enc_e2e_vs_nemo_validT_only"] = metrics(nemo_enc[:, :, :L], cml_enc[:, :, :L])
            entry["enc_isolated_vs_nemo_validT_only"] = metrics(nemo_enc[:, :, :L], cml_enc_nemomel[:, :, :L])
            # transpose check
            entry["enc_e2e_vs_nemo_transposed"] = metrics(
                nemo_enc, np.transpose(cml_enc, (0, 2, 1)) if cml_enc.shape[1] == nemo_enc.shape[2] else cml_enc
            )
        report[tag] = entry
        print(f"[{tag}] enc {cml_enc.shape} coreml_enc_len={entry['coreml_enc_len']}")
        print(f"   e2e full-T rel_l2 = {entry['enc_e2e_vs_nemo_full_T'].get('rel_l2')}")
        print(f"   isolated full-T rel_l2 = {entry['enc_isolated_vs_nemo_full_T'].get('rel_l2')}")
        if "enc_e2e_vs_nemo_validT_only" in entry:
            print(f"   e2e valid-T({nemo_enc_len}) rel_l2 = {entry['enc_e2e_vs_nemo_validT_only'].get('rel_l2')}")
            print(f"   isolated valid-T({nemo_enc_len}) rel_l2 = {entry['enc_isolated_vs_nemo_validT_only'].get('rel_l2')}")

    # shipped 6-bit baseline
    bp = BASE_DIR / "Preprocessor.mlmodelc"
    be = BASE_DIR / "Encoder.mlmodelc"
    if bp.exists() and be.exists():
        try:
            b_mel, b_enc, b_enc_len = coreml_mel_and_enc(bp, be, audio)
            base = {
                "enc_shape": list(b_enc.shape),
                "coreml_enc_len": (b_enc_len.tolist() if b_enc_len is not None else None),
                "mel_vs_nemo": metrics(nemo_mel, b_mel),
                "enc_e2e_vs_nemo_full_T": metrics(nemo_enc, b_enc),
            }
            if b_enc.shape == nemo_enc.shape:
                base["enc_e2e_vs_nemo_validT_only"] = metrics(nemo_enc[:, :, :nemo_enc_len], b_enc[:, :, :nemo_enc_len])
            report["shipped_6bit"] = base
            print(f"[shipped_6bit] enc {b_enc.shape} rel_l2 full-T = {base['enc_e2e_vs_nemo_full_T'].get('rel_l2')}")
        except Exception as e:
            report["shipped_6bit"] = {"error": repr(e)}
            print(f"[shipped_6bit] ERROR {e!r}")

    Path("./bench_results").mkdir(exist_ok=True)
    Path("./bench_results/diag_encoder_parity.json").write_text(json.dumps(report, indent=2))
    print("\nwrote ./bench_results/diag_encoder_parity.json")


if __name__ == "__main__":
    main()
