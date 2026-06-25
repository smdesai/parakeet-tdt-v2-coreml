# parakeet-transcribe

Standalone Swift command-line transcriber for **NVIDIA Parakeet-TDT-v2 (0.6B)** on
the fixed-shape CoreML encoder. Takes an audio file, prints the transcription.

It implements **Strategy C** — overlapping 15 s windows + one continuous TDT decode —
the long-form strategy validated in the Python ablation (`../longform_transcribe.py`,
`../exp_carried_state.py`). On the `lsc` set (9 clips, 31–66 s) this Swift port scores
**WER 0.0718**, matching the Python reference (0.0726) and closing on the NeMo ceiling
(0.0667), while the naive per-window baseline (Strategy A) scores 0.0864.

This is a pure-Swift port: no Python, no NeMo, no FluidAudio runtime — only CoreML +
AVFoundation from the system SDK.

## Build & run

```bash
swift build -c release
.build/release/parakeet-transcribe \
    --models ../parakeet_coreml_v2_final \
    --audio  /path/to/clip.wav
```

The runtime models are produced by the four‑step recipe in the
[top‑level README](../README.md) (convert FP16 → quantize encoder INT8 → compile →
add vocab).

The `--models` directory must contain the four runtime models plus the vocab:

```
parakeet_preprocessor.mlmodelc
parakeet_encoder.mlmodelc
parakeet_decoder.mlmodelc
parakeet_joint_decision_single_step.mlmodelc
parakeet_vocab.json
```

### Options

| Flag           | Default | Meaning                                                        |
|----------------|---------|----------------------------------------------------------------|
| `--models`     | —       | Directory with the 4 `.mlmodelc` + `parakeet_vocab.json`       |
| `--audio`      | —       | Input audio (any AVFoundation-decodable format; auto-resampled)|
| `--ctx-s`      | `2.5`   | Context seconds read each side of a window (encoded, not emitted) |
| `--baseline`   | off     | Use Strategy A (per-window decode + join) instead of C         |
| `--compute`    | `all`   | Compute units: `all` \| `cpu` \| `cpugpu` \| `ane`             |

## How it works (Strategy C)

1. **Load & resample** audio to mono Float32 @ 16 kHz (`AudioLoader`, AVAudioConverter).
2. **Plan windows** (`WindowPlanner`): slide a 15 s window whose 10 s *center* tiles the
   audio exactly. Each window also reads `--ctx-s` of look-around on both sides — encoded
   but **not** emitted. The first window has no left context, the last no right context
   (same as NeMo sees at true boundaries). Audio ≤ 15 s is a single full window.
3. **Encode + slice + concat** (`Encoder`): each window → preprocessor → encoder; map the
   center sample region to encoder frames via `ratio = validFrames / trueLen`; extract that
   frame slice. Concatenate all center slices into one `[1, 1024, ΣT]` stream (single alloc).
4. **One continuous TDT decode** (`TdtDecoder`) over the whole concatenated stream — never
   restarted mid-utterance, so no tokens are dropped at window seams.
5. **Detokenize** (`ParakeetTokenizer`): map text token ids → string, `▁` → space.

Both ingredients are required. Overlap alone (per-window decode) drops seam-straddling
tokens (Python WER 0.2715); continuous decode without overlap truncates seam frames
(0.0932). Together: 0.0726.

### Why TDT, not plain RNN-T

Each joint step predicts **a token AND a duration** (frames to skip). The duration advances
the time pointer on *every* step — blank and non-blank alike. The force-blank guard counts
emissions **per frame** (not globally): it only fires when too many tokens land on the same
frame, never skipping a frame mid-utterance during dense speech. (An earlier global counter
caused multi-word drops on dense passages — see `TdtDecoder.swift`.)

## Source map

| File                     | Responsibility                                                    |
|--------------------------|-------------------------------------------------------------------|
| `Const.swift`            | Verified model constants (shapes, vocab/blank, durations)         |
| `AudioLoader.swift`      | Decode + resample any input to mono Float32 @ 16 kHz              |
| `WindowPlanner.swift`    | Pure window tiling (centers tile `[0, n)` contiguously)           |
| `MLArray.swift`          | Stride-aware fp16/fp32 MLMultiArray read/write helpers            |
| `ModelRunner.swift`      | Loads the 4 models; typed wrappers for each prediction            |
| `Encoder.swift`          | Per-window encode → center-frame slice → concat into one stream   |
| `TdtDecoder.swift`       | Single continuous TDT decode loop                                 |
| `ParakeetTokenizer.swift`| Token-id → text                                                   |
| `Transcriber.swift`      | Strategy C (`transcribe`) and Strategy A (`transcribeBaseline`)   |
| `main.swift`             | Arg parsing + wiring                                              |

Full design rationale and the verified I/O contracts are in
[`../docs/swift-port-spec.md`](../docs/swift-port-spec.md).

## Vocabulary

`parakeet_vocab.json` is extracted from the `.nemo` archive's `*_tokenizer.vocab`
(1024 clean entries, ids 0–1023; `<unk>`@0; SentencePiece `▁` = leading space). The
blank id is 1024 (== `vocabSize`) and is never rendered. **Do not** use FluidAudio's
dumped vocab (1031 entries): its ids 1024–1030 map to real word-pieces, so blank would
render as text (`1024 → "▁warming"`). The tokenizer also defensively drops any id ≥ 1024.

## Tests

```bash
swift test
```

`WindowPlannerTests` checks the tiling invariants (contiguous centers, full coverage,
no overlap, no left-pad on the first window, no right-pad on the last, 10 s center width)
across n ∈ {1 s, 15 s, 15.0001 s, 31 s, 66 s}.

## Parity & numerical notes

Validated against the Python reference (`../longform_transcribe.py`) on the `lsc` set:

- **5 / 9 clips byte-for-byte identical** out of the box.
- With identical 16 kHz input (factoring out the resampler), **7 / 9 identical** — the
  remaining differences trace to `torchaudio`'s sinc resampler vs `AVAudioConverter`.
- The last 1–2 diffs are single-token argmax flips on **coined proper nouns** (e.g.
  "Xtandi" → "Tandy"/"Tandi") and a dropped function word — sub-threshold encoder
  numerical noise (CPU fp32 vs ANE fp16), not a logic divergence. They wash out in
  aggregate: Swift WER 0.0718 ≈ Python 0.0726.

The harmless `E5RT … slice_by_index: zero shape error` line on stderr/stdout is CoreML
teardown noise (appears in the Python path too); pipe stderr to `/dev/null` to suppress.

### Empty-output → CPU retry

On the default `.all` path, ~1 in 1,400 real clips can collapse to an **empty**
transcript: when speech begins softly after leading silence, the onset frame's
first-token-vs-blank decision is a near-tie, and encoder fp16 rounding (ANE *and*
GPU both flip; only CPU fp32 holds) tips it to blank — which, with the predictor
frozen at SOS until the first emission, cascades into an all-blank decode. The
transcriber detects the empty result and **retries that clip once on a `CPU_ONLY`
runner** (built lazily, reused). It triggers only on the rare empty case, so ANE
speed is preserved otherwise. Full analysis: [`docs/ane-onset-collapse.md`](docs/ane-onset-collapse.md).
