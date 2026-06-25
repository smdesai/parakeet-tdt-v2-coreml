#!/usr/bin/env python3
"""
Accuracy validation harness for Parakeet-TDT-v2 CoreML builds.

Compares encoder tensor parity and end-to-end WER across:
- NeMo PyTorch reference (ground truth)
- Shipped 6-bit baseline (FluidAudio)
- New FP32 build (if available)
- New FP16 build (if available)

Methodology:
1. Get NeMo reference transcription (ground truth text)
2. Encoder tensor parity: compare CoreML encoder outputs vs NeMo encoder output
3. End-to-end WER via encoder-swap: feed CoreML encoder output into NeMo decoder
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import coremltools as ct
from jiwer import wer


def load_nemo_model(nemo_path: str):
    """Load NeMo model from .nemo file."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo model from {nemo_path}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        nemo_path, map_location="cpu"
    )
    model.eval()
    print("NeMo model loaded successfully")
    return model


def get_nemo_reference_transcription(model, audio_paths: List[str]) -> Dict[str, str]:
    """Get reference transcriptions from NeMo PyTorch model."""
    print("\n=== Getting NeMo reference transcriptions ===")
    results = {}

    for audio_path in audio_paths:
        print(f"Transcribing {audio_path}...")
        transcription = model.transcribe([audio_path])[0]
        # NeMo returns a Hypothesis object, extract text
        if hasattr(transcription, 'text'):
            text = transcription.text
        else:
            text = str(transcription)
        results[audio_path] = text
        print(f"  Reference: {text}")

    return results


def get_nemo_encoder_output(model, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get encoder output from NeMo model.

    Returns:
        encoder_output: [B, D, T] numpy array
        encoded_lengths: [B] numpy array
        mel_features: [B, F, T] numpy array (preprocessor output)
    """
    print(f"\n=== Getting NeMo encoder output for {audio_path} ===")

    # Use NeMo's preprocessor to get the features
    with torch.no_grad():
        # Load audio via NeMo's audio loading
        import soundfile as sf
        audio, sr = sf.read(audio_path)

        # NeMo expects list of audio arrays
        # Get processed audio features
        processed_signal, processed_signal_len = model.preprocessor(
            input_signal=torch.tensor(audio).unsqueeze(0).float(),
            length=torch.tensor([len(audio)])
        )

        # Run through encoder
        encoder_output, encoded_lengths = model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_len
        )

        # Convert to numpy
        encoder_output_np = encoder_output.cpu().numpy()
        encoded_lengths_np = encoded_lengths.cpu().numpy()
        mel_features_np = processed_signal.cpu().numpy()

        print(f"  Mel features shape: {mel_features_np.shape}")
        print(f"  Encoder output shape: {encoder_output_np.shape}")
        print(f"  Encoded lengths: {encoded_lengths_np}")

        return encoder_output_np, encoded_lengths_np, mel_features_np


def get_compiled_model_io_names(model_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract input and output names from a compiled model's metadata.json.

    Returns:
        (input_names, output_names)
    """
    metadata_path = Path(model_path) / "metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: No metadata.json found at {metadata_path}")
        return [], []

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # metadata is a list, take first element
    if isinstance(metadata, list):
        metadata = metadata[0]

    input_names = [inp["name"] for inp in metadata.get("inputSchema", [])]
    output_names = [out["name"] for out in metadata.get("outputSchema", [])]

    return input_names, output_names


def load_coreml_model(model_path: str, compute_units: str = "CPU_ONLY"):
    """Load CoreML model (.mlmodelc or .mlpackage)."""
    print(f"Loading CoreML model from {model_path} (compute_units={compute_units})...")

    # Map compute units string to coremltools enum
    cu_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "ALL": ct.ComputeUnit.ALL,
    }
    cu = cu_map.get(compute_units, ct.ComputeUnit.CPU_ONLY)

    # Check if this is a compiled model (.mlmodelc) or package (.mlpackage)
    model_path_obj = Path(model_path)
    if model_path_obj.suffix == ".mlmodelc":
        # Use CompiledMLModel for .mlmodelc bundles
        print(f"  Loading compiled model (.mlmodelc)...")
        model = ct.models.CompiledMLModel(model_path, compute_units=cu)
        print(f"  Model loaded: {model_path}")

        # Get input/output names from metadata.json
        input_names, output_names = get_compiled_model_io_names(model_path)
        print(f"  Inputs: {input_names}")
        print(f"  Outputs: {output_names}")

        return model
    else:
        # Use MLModel for .mlpackage bundles
        model = ct.models.MLModel(model_path, compute_units=cu)
        print(f"  Model loaded: {model_path}")

        # Print spec for debugging
        spec = model.get_spec()
        print(f"  Inputs: {[inp.name for inp in spec.description.input]}")
        print(f"  Outputs: {[out.name for out in spec.description.output]}")

        return model


def get_coreml_encoder_output(
    preprocessor_path: str,
    encoder_path: str,
    audio_path: str,
    compute_units: str = "CPU_ONLY"
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Get encoder output from CoreML preprocessor + encoder.

    Returns:
        encoder_output: numpy array
        encoded_lengths: numpy array or None
        mel_features: numpy array (preprocessor output)
    """
    print(f"\n=== Getting CoreML encoder output ({encoder_path}) ===")

    # Load audio
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    print(f"  Audio shape: {audio.shape}, sample_rate: {sr}")

    # Load preprocessor
    preprocessor = load_coreml_model(preprocessor_path, compute_units)

    # Get preprocessor input/output names
    if hasattr(preprocessor, 'get_spec'):
        # MLModel (uncompiled)
        prep_spec = preprocessor.get_spec()
        prep_input_names = [inp.name for inp in prep_spec.description.input]
        prep_output_names = [out.name for out in prep_spec.description.output]
    else:
        # CompiledMLModel
        prep_input_names, prep_output_names = get_compiled_model_io_names(preprocessor_path)

    prep_input_name = prep_input_names[0]  # Primary input (audio)
    prep_output_name = prep_output_names[0]  # Primary output (mel)

    print(f"  Preprocessor input: {prep_input_name}, output: {prep_output_name}")

    # Run preprocessor
    # CoreML expects shape [1, num_samples] for audio
    audio_input = audio.astype(np.float32).reshape(1, -1)
    audio_length = np.array([len(audio)], dtype=np.int32)

    # Check if preprocessor needs audio_length
    prep_inputs = {prep_input_name: audio_input}
    if "audio_length" in prep_input_names:
        prep_inputs["audio_length"] = audio_length
        print(f"  Providing audio_length: {audio_length}")

    prep_output = preprocessor.predict(prep_inputs)
    mel = prep_output[prep_output_name]
    print(f"  Preprocessor output shape: {mel.shape}")

    # Load encoder
    encoder = load_coreml_model(encoder_path, compute_units)

    # Get encoder input/output names
    if hasattr(encoder, 'get_spec'):
        # MLModel (uncompiled)
        enc_spec = encoder.get_spec()
        enc_input_names = [inp.name for inp in enc_spec.description.input]
        enc_output_names = [out.name for out in enc_spec.description.output]
    else:
        # CompiledMLModel
        enc_input_names, enc_output_names = get_compiled_model_io_names(encoder_path)

    enc_input_name = enc_input_names[0]  # Primary input (mel)
    enc_output_name = enc_output_names[0]  # Primary output (encoder)

    print(f"  Encoder input: {enc_input_name}, output: {enc_output_name}")

    # Prepare encoder inputs
    enc_inputs = {enc_input_name: mel}

    # Check if encoder needs mel_length
    if "mel_length" in enc_input_names:
        # Get mel_length from preprocessor output if available
        if "mel_length" in prep_output:
            mel_length = prep_output["mel_length"]
        else:
            # Compute mel_length from mel shape
            mel_length = np.array([mel.shape[2]], dtype=np.int32)
        enc_inputs["mel_length"] = mel_length
        print(f"  Providing mel_length: {mel_length}")

    # Run encoder
    enc_output = encoder.predict(enc_inputs)
    encoder_output = enc_output[enc_output_name]
    print(f"  Encoder output shape: {encoder_output.shape}")

    # Return encoder output, encoded_lengths, and mel features
    return encoder_output, None, mel


def align_encoder_outputs(reference: np.ndarray, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align encoder output tensors for comparison.

    Expected format: [B, D, T] where B=batch, D=feature_dim (1024), T=time_frames

    Returns:
        aligned_reference, aligned_prediction
    """
    print(f"  Aligning encoder outputs:")
    print(f"    Reference shape: {reference.shape}")
    print(f"    Prediction shape: {prediction.shape}")

    # Check if shapes are already aligned
    if reference.shape == prediction.shape:
        print(f"    Shapes already match: {reference.shape}")
        return reference, prediction

    # Check for transpose issue: [B, D, T] vs [B, T, D]
    if len(reference.shape) == 3 and len(prediction.shape) == 3:
        if reference.shape[0] == prediction.shape[0]:  # Same batch size
            # Check if one is transposed
            if reference.shape[1] == prediction.shape[2] and reference.shape[2] == prediction.shape[1]:
                print(f"    WARNING: Detected transpose! Transposing prediction from {prediction.shape}")
                prediction = np.transpose(prediction, (0, 2, 1))
                print(f"    After transpose: {prediction.shape}")
                return reference, prediction

            # Check if time dimensions differ but feature dims match
            if reference.shape[1] == prediction.shape[1]:  # Same feature dimension
                min_t = min(reference.shape[2], prediction.shape[2])
                print(f"    Truncating time dimension to {min_t}")
                reference = reference[:, :, :min_t]
                prediction = prediction[:, :, :min_t]
                return reference, prediction

    print(f"    ERROR: Cannot align shapes {reference.shape} and {prediction.shape}")
    return reference, prediction


def compute_tensor_metrics(reference: np.ndarray, prediction: np.ndarray, align_first: bool = True) -> Dict[str, float]:
    """
    Compute tensor comparison metrics.

    Args:
        reference: Reference tensor
        prediction: Prediction tensor
        align_first: If True, attempt to align tensors first

    Returns:
        dict with max_abs_diff, mean_abs_diff, rel_l2, plus debug info
    """
    # Align tensors if requested
    if align_first:
        reference, prediction = align_encoder_outputs(reference, prediction)

    # Ensure shapes match after alignment
    if reference.shape != prediction.shape:
        print(f"  ERROR: Shape mismatch after alignment: ref {reference.shape} vs pred {prediction.shape}")
        return {
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "rel_l2": float("nan"),
            "error": f"Shape mismatch: {reference.shape} vs {prediction.shape}"
        }

    diff = prediction - reference
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))

    # Relative L2 norm
    ref_norm = np.linalg.norm(reference)
    diff_norm = np.linalg.norm(diff)
    pred_norm = np.linalg.norm(prediction)
    rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else float("nan")

    # Additional debug metrics
    print(f"  Metrics: ref_norm={ref_norm:.4f}, pred_norm={pred_norm:.4f}, diff_norm={diff_norm:.4f}")
    print(f"  Metrics: ref_mean={np.mean(reference):.4f}, pred_mean={np.mean(prediction):.4f}")
    print(f"  Metrics: ref_std={np.std(reference):.4f}, pred_std={np.std(prediction):.4f}")
    print(f"  Metrics: max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}, rel_l2={rel_l2:.6f}")

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rel_l2": rel_l2,
        "ref_norm": float(ref_norm),
        "pred_norm": float(pred_norm),
        "diff_norm": float(diff_norm),
    }


def decode_with_nemo(model, encoder_output: np.ndarray, encoded_lengths: np.ndarray) -> str:
    """
    Decode encoder output using NeMo's RNNT decoder.

    Args:
        model: NeMo model
        encoder_output: [B, D, T] encoder output
        encoded_lengths: [B] encoded lengths

    Returns:
        transcription: decoded text
    """
    print("\n=== Decoding with NeMo RNNT decoder ===")

    # Convert to torch tensors
    encoder_output_torch = torch.from_numpy(encoder_output).float()
    encoded_lengths_torch = torch.from_numpy(encoded_lengths).long()

    print(f"  Encoder output shape: {encoder_output_torch.shape}")
    print(f"  Encoded lengths: {encoded_lengths_torch}")

    # Use NeMo's decoding
    with torch.no_grad():
        # Try different API variations
        if hasattr(model.decoding, 'rnnt_decoder_predictions_tensor'):
            # Method 1: rnnt_decoder_predictions_tensor
            try:
                predictions = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output_torch, encoded_lengths_torch
                )
                # predictions is a tuple of (predictions, lengths)
                pred_ids = predictions[0][0].cpu().numpy()  # [B, T] -> [T]
                transcription = model.decoding.decode_tokens_to_str(pred_ids)
                print(f"  Decoded (method 1): {transcription}")
                return transcription
            except Exception as e:
                print(f"  Method 1 failed: {e}")

        # Method 2: Use the full decode method
        if hasattr(model, 'decoding') and hasattr(model.decoding, 'rnnt_decoder_predictions_tensor'):
            try:
                # Prepare encoder output for decoding
                # Some models expect encoder_output in specific format
                best_hyp = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output_torch, encoded_lengths_torch, return_hypotheses=True
                )
                print(f"  Decoded (method 2): {best_hyp}")

                # Extract text from hypothesis
                if isinstance(best_hyp, list) and len(best_hyp) > 0:
                    # List of Hypothesis objects
                    if hasattr(best_hyp[0], 'text'):
                        transcription = best_hyp[0].text
                    else:
                        transcription = str(best_hyp[0])
                elif hasattr(best_hyp, 'text'):
                    transcription = best_hyp.text
                elif isinstance(best_hyp, tuple):
                    best_hyp = best_hyp[0]
                    if isinstance(best_hyp, torch.Tensor):
                        transcription = model.decoding.decode_tokens_to_str(best_hyp[0].cpu().numpy())
                    else:
                        transcription = str(best_hyp)
                else:
                    transcription = str(best_hyp)

                print(f"  Decoded text: {transcription}")
                return transcription
            except Exception as e:
                print(f"  Method 2 failed: {e}")
                import traceback
                traceback.print_exc()

        # Method 3: Direct greedy decode using the decoder + joint
        try:
            # This is the most direct approach
            from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

            # Greedy decode using the model's greedy decoder
            if hasattr(model, 'decoding'):
                # Use the decoding strategy configured in the model
                # Pass encoder output through decoding
                hypotheses = []
                for b in range(encoder_output_torch.shape[0]):
                    # Get this batch's encoder output
                    enc_out_b = encoder_output_torch[b:b+1, :, :encoded_lengths_torch[b]]

                    # Initialize decoder state
                    state = model.decoding.decoder.initialize_state(enc_out_b)

                    # Greedy decode
                    hypothesis = Hypothesis(score=0.0, y_sequence=[], dec_state=state)

                    for t in range(enc_out_b.shape[2]):
                        # Get encoder output at time t
                        enc_t = enc_out_b[:, :, t:t+1]  # [1, D, 1]

                        # Run decoder
                        dec_out, state = model.decoding.decoder.predict(
                            hypothesis.y_sequence[-1] if hypothesis.y_sequence else None,
                            state,
                            add_sos=False
                        )

                        # Run joint
                        joint_out = model.joint.joint(enc_t, dec_out)  # [1, 1, 1, vocab_size]

                        # Get best token
                        logits = joint_out.squeeze()  # [vocab_size]
                        token = torch.argmax(logits).item()

                        # If not blank, add to hypothesis
                        if token != model.decoding.blank_id:
                            hypothesis.y_sequence.append(token)

                    hypotheses.append(hypothesis)

                # Convert to text
                transcription = model.decoding.decode_tokens_to_str(hypotheses[0].y_sequence)
                print(f"  Decoded (method 3): {transcription}")
                return transcription
        except Exception as e:
            print(f"  Method 3 failed: {e}")
            import traceback
            traceback.print_exc()

    raise RuntimeError("All NeMo decoding methods failed. Cannot decode encoder output.")


def compute_wer_metric(reference: str, hypothesis: str) -> float:
    """Compute WER using jiwer."""
    return wer(reference, hypothesis)


def validate_variant(
    variant_name: str,
    preprocessor_path: str,
    encoder_path: str,
    audio_path: str,
    nemo_model,
    nemo_encoder_output: np.ndarray,
    nemo_encoded_lengths: np.ndarray,
    nemo_mel_features: np.ndarray,
    reference_text: str,
    compute_units: str = "CPU_ONLY"
) -> Dict:
    """
    Validate a single variant (CoreML build).

    Returns:
        dict with mel_metrics, encoder_metrics, wer, transcription
    """
    print(f"\n{'='*60}")
    print(f"Validating variant: {variant_name}")
    print(f"{'='*60}")

    result = {
        "variant": variant_name,
        "mel_metrics": {},
        "encoder_metrics": {},
        "wer": None,
        "transcription": None,
        "error": None,
    }

    try:
        # Get CoreML encoder output
        coreml_encoder_output, coreml_encoded_lengths, coreml_mel_features = get_coreml_encoder_output(
            preprocessor_path, encoder_path, audio_path, compute_units
        )

        # Compute mel features parity
        print("  Comparing mel features (preprocessor output):")
        result["mel_metrics"] = compute_tensor_metrics(
            nemo_mel_features, coreml_mel_features
        )

        # Compute encoder tensor metrics
        print("  Comparing encoder outputs:")
        result["encoder_metrics"] = compute_tensor_metrics(
            nemo_encoder_output, coreml_encoder_output
        )
        print(f"  Encoder metrics: {result['encoder_metrics']}")

        # Decode with NeMo
        # Use NeMo's encoded_lengths (CoreML doesn't provide it)
        try:
            transcription = decode_with_nemo(
                nemo_model, coreml_encoder_output, nemo_encoded_lengths
            )
            result["transcription"] = transcription

            # Compute WER
            wer_score = compute_wer_metric(reference_text, transcription)
            result["wer"] = wer_score
            print(f"  WER: {wer_score:.4f}")
            print(f"  Reference: {reference_text}")
            print(f"  Hypothesis: {transcription}")
        except Exception as e:
            result["error"] = f"Decoding failed: {str(e)}"
            print(f"  ERROR: {result['error']}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {result['error']}")
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Parakeet-TDT-v2 CoreML accuracy")
    parser.add_argument(
        "--nemo-path",
        type=str,
        default="/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo",
        help="Path to NeMo .nemo file"
    )
    parser.add_argument(
        "--audio-15s",
        type=str,
        default="./audio/yc_first_minute_16k_15s.wav",
        help="Path to 15s audio file"
    )
    parser.add_argument(
        "--audio-full",
        type=str,
        default="./audio/yc_first_minute_16k.wav",
        help="Path to full minute audio file"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="/Users/sdesai/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2/",
        help="Path to shipped 6-bit baseline directory"
    )
    parser.add_argument(
        "--fp32-dir",
        type=str,
        default="./parakeet_coreml_v2_fp32",
        help="Path to FP32 build directory"
    )
    parser.add_argument(
        "--fp16-dir",
        type=str,
        default="./parakeet_coreml_v2_fp16",
        help="Path to FP16 build directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bench_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="CPU_ONLY",
        choices=["CPU_ONLY", "CPU_AND_GPU", "ALL"],
        help="CoreML compute units"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load NeMo model
    nemo_model = load_nemo_model(args.nemo_path)

    # Phase 1: Get NeMo reference
    print("\n" + "="*60)
    print("PHASE 1: NeMo Reference")
    print("="*60)

    # Get reference transcriptions
    reference_texts = get_nemo_reference_transcription(
        nemo_model, [args.audio_15s, args.audio_full]
    )
    reference_text_15s = reference_texts[args.audio_15s]

    # Get NeMo encoder output for 15s clip
    nemo_encoder_output, nemo_encoded_lengths, nemo_mel_features = get_nemo_encoder_output(
        nemo_model, args.audio_15s
    )

    # Save NeMo reference
    nemo_reference = {
        "variant": "NeMo PyTorch (reference)",
        "transcription_15s": reference_text_15s,
        "transcription_full": reference_texts[args.audio_full],
        "encoder_output_shape": list(nemo_encoder_output.shape),
        "encoded_lengths": nemo_encoded_lengths.tolist(),
        "encoder_metrics": {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "rel_l2": 0.0,
        },
        "wer": 0.0,  # Reference is always 0 WER
    }

    # Save NeMo reference results
    nemo_output_path = os.path.join(args.output_dir, "v2_accuracy_nemo_reference.json")
    with open(nemo_output_path, "w") as f:
        json.dump(nemo_reference, f, indent=2)
    print(f"\nSaved NeMo reference to {nemo_output_path}")

    # Phase 2: Validate variants
    print("\n" + "="*60)
    print("PHASE 2: Validate CoreML Variants")
    print("="*60)

    variants = []

    # Shipped 6-bit baseline
    baseline_preprocessor = os.path.join(args.baseline_dir, "Preprocessor.mlmodelc")
    baseline_encoder = os.path.join(args.baseline_dir, "Encoder.mlmodelc")

    if os.path.exists(baseline_preprocessor) and os.path.exists(baseline_encoder):
        baseline_result = validate_variant(
            "Shipped 6-bit baseline",
            baseline_preprocessor,
            baseline_encoder,
            args.audio_15s,
            nemo_model,
            nemo_encoder_output,
            nemo_encoded_lengths,
            nemo_mel_features,
            reference_text_15s,
            args.compute_units
        )
        variants.append(baseline_result)

        # Save baseline results
        baseline_output_path = os.path.join(args.output_dir, "v2_accuracy_shipped_6bit.json")
        with open(baseline_output_path, "w") as f:
            json.dump(baseline_result, f, indent=2)
        print(f"\nSaved baseline results to {baseline_output_path}")
    else:
        print(f"WARNING: Baseline not found at {args.baseline_dir}")

    # New FP32 build
    fp32_preprocessor = os.path.join(args.fp32_dir, "parakeet_preprocessor.mlpackage")
    fp32_encoder = os.path.join(args.fp32_dir, "parakeet_encoder.mlpackage")

    if os.path.exists(fp32_preprocessor) and os.path.exists(fp32_encoder):
        fp32_result = validate_variant(
            "New FP32",
            fp32_preprocessor,
            fp32_encoder,
            args.audio_15s,
            nemo_model,
            nemo_encoder_output,
            nemo_encoded_lengths,
            nemo_mel_features,
            reference_text_15s,
            args.compute_units
        )
        variants.append(fp32_result)

        # Save FP32 results
        fp32_output_path = os.path.join(args.output_dir, "v2_accuracy_fp32.json")
        with open(fp32_output_path, "w") as f:
            json.dump(fp32_result, f, indent=2)
        print(f"\nSaved FP32 results to {fp32_output_path}")
    else:
        print(f"SKIP: FP32 build not found at {args.fp32_dir}")
        print(f"  Looking for: {fp32_preprocessor}, {fp32_encoder}")

    # New FP16 build
    fp16_preprocessor = os.path.join(args.fp16_dir, "parakeet_preprocessor.mlpackage")
    fp16_encoder = os.path.join(args.fp16_dir, "parakeet_encoder.mlpackage")

    if os.path.exists(fp16_preprocessor) and os.path.exists(fp16_encoder):
        fp16_result = validate_variant(
            "New FP16",
            fp16_preprocessor,
            fp16_encoder,
            args.audio_15s,
            nemo_model,
            nemo_encoder_output,
            nemo_encoded_lengths,
            nemo_mel_features,
            reference_text_15s,
            args.compute_units
        )
        variants.append(fp16_result)

        # Save FP16 results
        fp16_output_path = os.path.join(args.output_dir, "v2_accuracy_fp16.json")
        with open(fp16_output_path, "w") as f:
            json.dump(fp16_result, f, indent=2)
        print(f"\nSaved FP16 results to {fp16_output_path}")
    else:
        print(f"SKIP: FP16 build not found at {args.fp16_dir}")

    # Final comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(f"{'Variant':<30} {'Encoder rel_L2':>15} {'Encoder max_abs':>15} {'WER':>10}")
    print("-" * 72)

    # NeMo reference
    print(f"{'NeMo PyTorch (reference)':<30} {0.0:>15.6f} {0.0:>15.6f} {0.0:>10.4f}")

    # Variants
    for variant in variants:
        name = variant["variant"]
        metrics = variant["encoder_metrics"]
        rel_l2 = metrics.get("rel_l2", float("nan"))
        max_abs = metrics.get("max_abs_diff", float("nan"))
        wer_val = variant.get("wer", float("nan"))

        if variant.get("error"):
            print(f"{name:<30} {'ERROR':>15} {'ERROR':>15} {'ERROR':>10}")
            print(f"  Error: {variant['error']}")
        else:
            print(f"{name:<30} {rel_l2:>15.6f} {max_abs:>15.6f} {wer_val:>10.4f}")

    # Save combined results
    combined_results = {
        "reference": nemo_reference,
        "variants": variants,
        "audio_15s": args.audio_15s,
        "audio_full": args.audio_full,
        "compute_units": args.compute_units,
    }

    combined_output_path = os.path.join(args.output_dir, "v2_accuracy_combined.json")
    with open(combined_output_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nSaved combined results to {combined_output_path}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Find baseline and new builds
    baseline = next((v for v in variants if "baseline" in v["variant"].lower()), None)
    fp32 = next((v for v in variants if "fp32" in v["variant"].lower()), None)
    fp16 = next((v for v in variants if "fp16" in v["variant"].lower()), None)

    if not baseline:
        print("INCOMPLETE: Baseline not measured")
    elif baseline.get("error"):
        print(f"BASELINE FAILED: {baseline['error']}")
    else:
        baseline_rel_l2 = baseline["encoder_metrics"]["rel_l2"]
        baseline_wer = baseline.get("wer", float("nan"))

        print(f"Baseline (6-bit): rel_L2={baseline_rel_l2:.6f}, WER={baseline_wer:.4f}")

        if fp32 and not fp32.get("error"):
            fp32_rel_l2 = fp32["encoder_metrics"]["rel_l2"]
            fp32_wer = fp32.get("wer", float("nan"))

            improvement_l2 = ((baseline_rel_l2 - fp32_rel_l2) / baseline_rel_l2) * 100
            improvement_wer = ((baseline_wer - fp32_wer) / baseline_wer) * 100 if baseline_wer > 0 else 0

            print(f"FP32: rel_L2={fp32_rel_l2:.6f} ({improvement_l2:+.1f}%), WER={fp32_wer:.4f} ({improvement_wer:+.1f}%)")

            if fp32_rel_l2 < baseline_rel_l2 and fp32_wer <= baseline_wer:
                print("✓ FP32 BEATS BASELINE on encoder parity and WER")
            elif fp32_rel_l2 < baseline_rel_l2:
                print("⚠ FP32 has better encoder parity but worse/equal WER")
            else:
                print("✗ FP32 does not beat baseline")

        if fp16 and not fp16.get("error"):
            fp16_rel_l2 = fp16["encoder_metrics"]["rel_l2"]
            fp16_wer = fp16.get("wer", float("nan"))

            improvement_l2 = ((baseline_rel_l2 - fp16_rel_l2) / baseline_rel_l2) * 100
            improvement_wer = ((baseline_wer - fp16_wer) / baseline_wer) * 100 if baseline_wer > 0 else 0

            print(f"FP16: rel_L2={fp16_rel_l2:.6f} ({improvement_l2:+.1f}%), WER={fp16_wer:.4f} ({improvement_wer:+.1f}%)")

            if fp16_rel_l2 < baseline_rel_l2 and fp16_wer <= baseline_wer:
                print("✓ FP16 BEATS BASELINE on encoder parity and WER")
            elif fp16_rel_l2 < baseline_rel_l2:
                print("⚠ FP16 has better encoder parity but worse/equal WER")
            else:
                print("✗ FP16 does not beat baseline")

    print("\n" + "="*60)
    print("Validation complete. Results saved to:")
    print(f"  {combined_output_path}")
    for variant in variants:
        variant_name = variant["variant"].lower().replace(" ", "_")
        variant_path = os.path.join(args.output_dir, f"v2_accuracy_{variant_name}.json")
        if os.path.exists(variant_path):
            print(f"  {variant_path}")


if __name__ == "__main__":
    main()
