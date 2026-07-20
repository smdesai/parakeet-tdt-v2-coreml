#!/usr/bin/env python3
"""Export ONLY the single-step joint TOKEN-LOGITS model for keyword boosting.

This is a targeted companion to convert-parakeet.py: it re-exports just
`parakeet_joint_logits_single_step.mlpackage` (the pre-argmax token-logits joint
that Swift-side keyword boosting needs) WITHOUT re-running the full 8-model
pipeline, so the deployed encoder/decoder/joint_decision artifacts are left
untouched.

It reuses the verified `JointWrapper` / `JointLogitsSingleStep` classes and the
`_coreml_convert` / `_save_mlpackage` / `ExportSettings` helpers from the existing
modules, so tracing/conversion is identical to convert-parakeet.py's own block.

Only the joint sub-module runs a forward pass here (on zero step tensors of the
correct shape); the encoder/preprocessor are not needed because the joint's
single-step inputs are fixed [1,1024,1] and [1,640,1].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
import coremltools as ct
import nemo.collections.asr as nemo_asr

from individual_components import (
    JointWrapper,
    JointLogitsSingleStep,
    ExportSettings,
    _coreml_convert,
)
# _save_mlpackage lives in convert-parakeet.py (hyphenated filename); import by path.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("convert_parakeet", Path(__file__).parent / "convert-parakeet.py")
_cp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cp)
_save_mlpackage = _cp._save_mlpackage

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def export(
    nemo_path: Path = typer.Option(..., "--nemo-path", exists=True, resolve_path=True,
                                   help="Path to parakeet-tdt-0.6b-v2 .nemo checkpoint"),
    output_dir: Path = typer.Option(Path("parakeet_coreml_v2_final"), "--output-dir",
                                    help="Where to write parakeet_joint_logits_single_step.mlpackage"),
) -> None:
    typer.echo(f"Loading NeMo model from {nemo_path}…")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location="cpu")
    asr_model.eval()

    vocab_size = int(asr_model.tokenizer.vocab_size)
    num_extra = int(asr_model.joint.num_extra_outputs)
    decoder_hidden = int(asr_model.decoder.pred_hidden)
    typer.echo(f"vocab_size={vocab_size} num_extra={num_extra} decoder_hidden={decoder_hidden}")

    joint = JointWrapper(asr_model.joint.eval())

    # Single-step reference tensors: encoder_step [1,1024,1], decoder_step [1,640,1].
    # Zeros are shape/dtype-correct for tracing; the traced graph is data-independent.
    enc_step = torch.zeros(1, int(asr_model.encoder._feat_out if hasattr(asr_model.encoder, "_feat_out") else 1024), 1, dtype=torch.float32)
    # Guard: joint enc projection expects encoder channel dim; force 1024 (Const.encoderChannels).
    if enc_step.shape[1] != 1024:
        enc_step = torch.zeros(1, 1024, 1, dtype=torch.float32)
    dec_step = torch.zeros(1, decoder_hidden, 1, dtype=torch.float32)

    export_settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        max_audio_seconds=15.0,
        max_symbol_steps=1,
    )

    typer.echo("Tracing single-step joint LOGITS…")
    jl = JointLogitsSingleStep(joint, vocab_size=vocab_size, num_extra=num_extra)
    with torch.inference_mode():
        traced = torch.jit.trace(jl, (enc_step, dec_step), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(name="encoder_step", shape=(1, enc_step.shape[1], 1), dtype=np.float32),
        ct.TensorType(name="decoder_step", shape=(1, dec_step.shape[1], 1), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="token_logits", dtype=np.float32),
        ct.TensorType(name="duration", dtype=np.int32),
    ]
    model = _coreml_convert(traced, inputs, outputs, export_settings,
                            compute_units_override=ct.ComputeUnit.CPU_ONLY)
    out_path = output_dir / "parakeet_joint_logits_single_step.mlpackage"
    _save_mlpackage(model, out_path, "Parakeet single-step joint TOKEN LOGITS (pre-argmax, for boosting)")
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
