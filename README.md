# Parakeet‑TDT v2 (0.6B) — CoreML Export, Quantization, and CLI

Tools to export NVIDIA **Parakeet‑TDT v2 (0.6B)** RNNT ASR to CoreML, quantize the
encoder to INT8 for ANE, validate numerical parity with the NeMo reference, and run a
standalone Swift transcriber. All CoreML components use a fixed **15‑second** audio
window for export and validation.

The deployable build is **not** a single command — it is a four‑step recipe:
**convert (FP16) → quantize encoder (INT8) → compile to `.mlmodelc` → assemble a runtime
dir**. Each step and its verification is documented below.

## Test environment

Measurements here were taken on an Apple **M4 Pro** (48 GB), macOS 14+, Xcode 16+
(`coremlcompiler` from the active Xcode). Python toolchain is pinned in `pyproject.toml`
/ `uv.lock` (Python 3.10, torch 2.7.0, coremltools 9.0b1, nemo‑toolkit 2.3.1).

## Prerequisites

- `uv` for Python dependency management.
- Xcode command‑line tools (`xcrun coremlcompiler` must resolve).
- The **`.nemo` checkpoint**: `parakeet-tdt-0.6b-v2.nemo`. Either download it ahead of
  time, or omit `--nemo-path` to let conversion fetch `nvidia/parakeet-tdt-0.6b-v2`.
  The examples below assume it is at `~/Downloads/parakeet-tdt-0.6b-v2.nemo`.

```bash
uv sync          # create .venv and install all Python deps
```

---

## Generate the CoreML model from scratch

The end result is a runtime directory holding **four** compiled models plus the vocab:

```
parakeet_encoder.mlmodelc                     # INT8, ~568 MB, ANE
parakeet_decoder.mlmodelc                     # FP16, ~14 MB,  CPU
parakeet_joint_decision_single_step.mlmodelc  # FP16, ~3.3 MB, CPU
parakeet_preprocessor.mlmodelc                # FP16, ~620 KB, CPU+GPU
parakeet_vocab.json                           # 1024 entries, from the .nemo
```

Set the checkpoint path once:

```bash
NEMO=~/Downloads/parakeet-tdt-0.6b-v2.nemo
```

### Step 1 — Convert to FP16 `.mlpackage`

Exports every sub‑module (preprocessor, encoder, decoder, joint, and the fused
mel+encoder / joint+decision variants) with the fixed 15 s contract, at **FLOAT16**
precision (half the size, ANE‑eligible).

```bash
uv run python convert-parakeet.py \
  --nemo-path "$NEMO" \
  --output-dir parakeet_coreml_v2_fp16 \
  --compute-precision FLOAT16
```

> **Note:** `convert-parakeet.py` is a **single‑command** tool — there is no `convert`
> subcommand. The default `--compute-precision` is `FLOAT32`; you must pass `FLOAT16`
> here. The output dir name `parakeet_coreml_v2_fp16` matters — the next step reads it
> by that exact name.

Result: `parakeet_coreml_v2_fp16/` with the FP16 encoder (~1.1 GB) and the small heads.

### Step 2 — Quantize the encoder to INT8

The encoder dominates size and is the only component that benefits from quantization.
`compress_encoder.py` applies coremltools INT8 linear quantization (per‑channel,
symmetric) **directly** to the FP16 `parakeet_encoder.mlpackage` — no NeMo round‑trip.

```bash
uv run python compress_encoder.py --scheme int8
```

- Reads `parakeet_coreml_v2_fp16/parakeet_encoder.mlpackage` (~1.19 GB).
- Writes `parakeet_coreml_v2_int8/parakeet_encoder.mlpackage` (~567 MB, **50%**), and
  copies the FP16 preprocessor alongside so the dir is a usable pair.

> Other schemes exist for experimentation (`--scheme palett8|palett6|palett4`, or
> `--all`). INT8 per‑channel is the shipped choice: it keeps NeMo drift near FP16's
> (~0.003) instead of the old 6‑bit baseline's (~0.076). See **Verification** below.

### Step 3 — Compile `.mlpackage` → `.mlmodelc`

The Swift runtime loads **compiled** `.mlmodelc` bundles, not `.mlpackage`. Compile the
four runtime models — INT8 encoder from step 2, the rest (FP16) from step 1 — into a
fresh runtime dir.

> `compile_modelc.py` is a convenience walker, but it hardcodes its source dirs
> (`parakeet_coreml`, `parakeet_coreml_quantized`) and will **not** pick up the
> `parakeet_coreml_v2_*` dirs. Compile the four packages directly with `xcrun`:

```bash
FRESH=parakeet_coreml_v2_final
mkdir -p "$FRESH"
COMPILER=$(xcrun --find coremlcompiler)

"$COMPILER" compile parakeet_coreml_v2_int8/parakeet_encoder.mlpackage                  "$FRESH"
"$COMPILER" compile parakeet_coreml_v2_fp16/parakeet_decoder.mlpackage                  "$FRESH"
"$COMPILER" compile parakeet_coreml_v2_fp16/parakeet_joint_decision_single_step.mlpackage "$FRESH"
"$COMPILER" compile parakeet_coreml_v2_fp16/parakeet_preprocessor.mlpackage             "$FRESH"
```

### Step 4 — Add the vocabulary (extract from the `.nemo`)

The vocab is **not** emitted by conversion — it is extracted from the `.nemo` archive's
`*_tokenizer.vocab` (a `.nemo` is just a tar of the checkpoint + tokenizer artifacts;
no NeMo install or model load needed). This produces the exact flat `{"id":"piece"}`
JSON the Swift tokenizer parses — **1024 entries, ids 0…1023**, `<unk>`@0, SentencePiece
`▁` (U+2581) leading‑space marker.

```bash
VOCAB=$(tar tf "$NEMO" | grep '_tokenizer.vocab$')          # e.g. a471…_tokenizer.vocab
tar xf "$NEMO" -O "$VOCAB" | python3 -c '
import sys, json
vocab = {str(i): line.rstrip("\n").split("\t")[0]
         for i, line in enumerate(sys.stdin)}
json.dump(vocab, open("parakeet_coreml_v2_final/parakeet_vocab.json", "w"),
          ensure_ascii=False, indent=0)
print("wrote", len(vocab), "tokens")                         # -> 1024
'
```

> **⚠ Do NOT use FluidAudio's `parakeet_vocab.json` verbatim.** It agrees with the
> `.nemo` on ids 0…1023 but appends 7 spurious rows 1024…1030; since `blank_id == 1024`,
> a detokenizer that looked up id 1024 would emit the word `"warming"` for every blank.
> The `.nemo` extraction (no row ≥ 1024) removes the footgun structurally.

`parakeet_coreml_v2_final/` is now the complete, deployable runtime dir.

---

## Build & run the CLI

The standalone Swift transcriber lives in `parakeet-transcribe/`. It implements
**Strategy C** (overlapping 15 s windows + one continuous TDT decode) and depends only
on CoreML + AVFoundation — no Python, NeMo, or FluidAudio runtime.

```bash
cd parakeet-transcribe
swift build -c release
.build/release/parakeet-transcribe \
  --models ../parakeet_coreml_v2_final \
  --audio  /path/to/clip.wav
```

Key flags (`--help` for all): `--models` (runtime dir from step 4), `--audio` (any
AVFoundation‑decodable file, auto‑resampled to 16 kHz mono), `--compute`
(`all`|`cpu`|`cpugpu`|`ane`, default `all`), `--baseline` (Strategy A instead of C).
The transcript prints to **stdout**; an RTFx/timing line prints to **stderr**.

See [`parakeet-transcribe/README.md`](parakeet-transcribe/README.md) for the full design
(window planning, TDT decode, the empty‑output→CPU‑fp32 retry, the source map) and
[`docs/swift-port-spec.md`](docs/swift-port-spec.md) for the verified I/O contracts.

---

## Verification

### A. Byte‑identical transcript gate (the primary CLI gate)

The transcriber's accuracy gate is an **exact‑match** content hash, not WER. On the
reference clip the clean transcript is **SHA `e93b6b90…`, 12 507 words**.

```bash
cd parakeet-transcribe
.build/release/parakeet-transcribe \
    --audio ~/Downloads/SEP-123-003b-md.mp3 \
    --models ../parakeet_coreml_v2_final 2>/dev/null \
  | sed -E 's/^.*zero shape error\.//' \
  | shasum
# => e93b6b90f0e5f1e792722c7da724c277ede75f0e   (12507 words)
```

> **CoreML stdout noise.** CoreML emits a harmless, non‑deterministic
> `E5RT … ios17.slice_by_index: zero shape error.` line onto **stdout**, glued to the
> transcript start (~19 words). It breaks naïve SHA/word‑count comparison — **strip it
> first** with the `sed` above. The RTFx report goes to **stderr** (`2>/dev/null` drops
> only the noise, not the transcript).

### B. Swift unit tests (window‑planner invariants)

```bash
cd parakeet-transcribe && swift test
```

### C. Encoder fidelity gate (quantization quality)

Before trusting a quantized encoder, confirm its NeMo drift stays near FP16's, not the
old 6‑bit baseline's. `gate_compressed.py` reuses the trusted dataset‑WER machinery and
reports encoder `rel_l2` vs NeMo plus WER / drift‑vs‑NeMo over a dataset subset
(requires the `.nemo` at the path in the script header).

```bash
uv run python gate_compressed.py --builds int8 --dataset fda --limit 80
```

Pass rule: INT8 drift should sit near FP16's (~0.003), well below the shipped 6‑bit's
(~0.076), and `rel_l2` ≪ 0.238. A focused single‑clip parity check is also available via
`diag_encoder_parity.py`.

---

## Repository map

| Path | Purpose |
|---|---|
| `convert-parakeet.py` | Step 1 — export all sub‑modules to CoreML (fixed 15 s window). |
| `compress_encoder.py` | Step 2 — INT8 / palettize the FP16 encoder `.mlpackage`. |
| `compile_modelc.py` | Helper to compile `.mlpackage` → `.mlmodelc` (hardcoded source dirs — see step 3). |
| `compare-components.py` | Torch‑vs‑CoreML parity + latency per component; writes plots. |
| `quantize_coreml.py` | Broad quantization sweep (size • quality • speed roll‑up). |
| `gate_compressed.py`, `gate_librispeech.py` | Accuracy gates for compressed candidates. |
| `diag_*.py` | Focused diagnostics (encoder parity, dataset WER, multi‑window). |
| `longform_transcribe.py`, `exp_carried_state.py` | Python long‑form (Strategy C) reference + ablation. |
| `parakeet-transcribe/` | Standalone Swift CLI transcriber (Strategy C). |
| `ParakeetTranscribeApp/` | iOS app wrapping the same Core pipeline. |
| `OPTIMIZATIONS.md` | Throughput work on the Swift transcriber (153×→340× Mac, 245× device). |
| `context/`, `docs/` | Architecture notes, conversion plan, and the Swift‑port spec. |

## Notes & limits

- Fixed 15‑second window shapes are required for all CoreML exports and validations.
- Minimum deployment target iOS 17 / macOS 14; models are MLProgram, ANE‑eligible when
  loaded with `ComputeUnits=ALL`. Conversion pins the preprocessor/mel‑encoder to
  `CPU_ONLY` for export (runtime compute units are chosen when loading).
- The intermediate dirs (`parakeet_coreml_v2_fp16/`, `parakeet_coreml_v2_int8/`) and the
  FP32 `parakeet_coreml/` are regenerable and gitignored; only the assembled
  `parakeet_coreml_v2_final/` is the runtime deliverable.

## Acknowledgements

- Parakeet‑TDT v2 model from NVIDIA NeMo (`nvidia/parakeet-tdt-0.6b-v2`).
- This directory provides export/quantization/validation utilities and a standalone
  Swift transcriber to reproduce quality and performance on Apple devices.
