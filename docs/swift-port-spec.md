# Standalone Swift Port — Parakeet-TDT-v2 Long-Form Transcriber

**Goal:** A self-contained SwiftPM command-line executable. User provides one audio
file; program prints the transcription.

```
$ parakeet-transcribe --models ./parakeet_coreml_v2_final --audio meeting.wav
the quarterly numbers came in ahead of forecast and the board …
```

It implements the validated **long-form strategy** for the fixed-15s CoreML
encoder: **overlapping windows + one continuous TDT decode** (Strategy C from the
`exp_carried_state.py` ablation; mirrors `longform_transcribe.py`). On lsc (9 clips,
31–66s) this scored WER **0.0726** vs NeMo's single-pass ceiling **0.0667** — the
two single-ingredient variants both fail (carried-no-overlap 0.0932; overlap-but-
per-window-decode 0.2715).

> **⚠ This is TDT, not plain RNN-T.** The existing `CoreMLParakeet.swift` in
> parakeet-unified is plain RNN-T (no duration head) and is **not reusable** for the
> decode loop. TDT predicts a token **and** a duration (how many frames to skip),
> which changes the inner loop. The canonical reference is FluidAudio's
> `TdtDecoderV3.swift` — this spec re-derives its semantics for a single continuous
> decode over concatenated center frames.

---

## 1. Inputs & artifacts

### 1.1 CoreML models (from `parakeet_coreml_v2_final/`, compiled `.mlmodelc`)

Only **four** models are needed for the runtime path. Verified I/O contracts
(`metadata.json` of each `.mlmodelc`):

| Model | Inputs | Outputs |
|---|---|---|
| `parakeet_preprocessor` | `audio_signal` f32 `[1,N]` (flex 1…240000) · `audio_length` i32 `[1]` | `mel` f32 `[1,128,T_mel]` · `mel_length` i32 `[1]` |
| `parakeet_encoder` | `mel` f32 `[1,128,1501]` · `mel_length` i32 `[1]` | `encoder` f32 `[1,1024,188]` · `encoder_length` i32 `[1]` |
| `parakeet_decoder` | `targets` i32 `[1,1]` · `target_length` i32 `[1]` · `h_in` f32 `[2,1,640]` · `c_in` f32 `[2,1,640]` | `decoder` f32 `[1,640,1]` · `h_out` f32 `[2,1,640]` · `c_out` f32 `[2,1,640]` |
| `parakeet_joint_decision_single_step` | `encoder_step` f32 `[1,1024,1]` · `decoder_step` f32 `[1,640,1]` | `token_id` i32 `[1,1,1]` · `token_prob` f32 `[1,1,1]` · `duration` i32 `[1,1,1]` |

The `joint_decision_single_step` model does the argmax internally (token over the
first 1025 logits; duration over the last 5) and returns the decoded `token_id` and
`duration` bin directly — so the Swift side does **no** logit argmax. (Raw-logit
`parakeet_joint` and `parakeet_joint_decision` exist too but are not used here.)

> **Encoder shape note.** The encoder input mel is **fixed** at `[1,128,1501]`. The
> preprocessor emits `mel_length` frames for the true (unpadded) audio; mel must be
> right-padded to 1501 columns before the encoder. `encoder_length` then reports the
> valid frame count (≈ `mel_length/8`, subsampling factor 8 → ≤188 frames for 15s).
> Trust `encoder_length`, never the static 188.

### 1.2 Vocabulary — extract from the `.nemo` (authoritative, clean)

**Source of truth = the `.nemo` file.** A `.nemo` is just a tar of the checkpoint plus
its tokenizer artifacts; the vocab can be pulled out directly — no model load, no GPU,
no NeMo install needed. This is the accuracy-first path and what the spec recommends.

✓ VERIFIED — the archive contains three tokenizer files; the relevant one is
`*_tokenizer.vocab`, which is **exactly 1024 lines (ids 0…1023)**, `<unk>` at id 0,
SentencePiece `▁` (U+2581) leading-space marker, tab-separated `piece<TAB>logprob`:

```bash
# extract → flat {"id":"piece"} JSON next to the models (no NeMo dependency)
NEMO=/Users/sdesai/Downloads/parakeet-tdt-0.6b-v2.nemo
VOCAB=$(tar tf "$NEMO" | grep '_tokenizer.vocab$')          # e.g. a471…_tokenizer.vocab
tar xf "$NEMO" -O "$VOCAB" | python3 -c '
import sys, json
vocab = {str(i): line.rstrip("\n").split("\t")[0]
         for i, line in enumerate(sys.stdin)}
json.dump(vocab, open("parakeet_coreml_v2_final/parakeet_vocab.json","w"),
          ensure_ascii=False, indent=0)
print("wrote", len(vocab), "tokens")                         # -> 1024
'
```

The result is the exact format `ParakeetTokenizer.swift` parses: flat
`{ "id": "piece" }`, string keys, `ensure_ascii=False`. Ids `0…1023` are the real text
tokens (`0 = <unk>`); **blank `1024` is intentionally absent** and is never rendered.

> **⚠ Do NOT use FluidAudio's `parakeet_vocab.json` verbatim.** A copy ships at
> `…/FluidAudio/Models/parakeet-tdt-0.6b-v2/parakeet_vocab.json`, and ✓ VERIFIED it
> **agrees with the `.nemo` on all of ids 0…1023 (zero mismatches)** — but it appends 7
> spurious rows `1024…1030` mapping to real word-pieces (`1024→"▁warming"`,
> `1025→"▁global"`, … `1030→"▁urge"`). Since `blank_id == 1024`, a naïve detokenizer
> that looks up id 1024 would emit the word "warming" for every blank.
>
> The `.nemo`-extracted vocab (1024 entries, no row ≥1024) makes this trap structurally
> impossible. If you must reuse the FA file instead, the §7 loop already filters
> `blankId` before §8 detokenize — but only the extraction above removes the footgun
> entirely. Either way the single-step joint argmaxes the token over the first 1025
> columns, so it can only ever emit ids `0…1023` (text) or `1024` (blank); 1025–1030 are
> never produced.

---

## 2. Constants (verified — do not change without re-deriving)

```swift
enum Const {
    static let sampleRate      = 16_000
    static let winSamples      = 240_000          // 15 s — fixed encoder window
    static let ctxSamples      = 40_000           // 2.5 s look-around each side
    static var centerSamples: Int { winSamples - 2 * ctxSamples }   // 160_000 (10 s)
    static let melBins         = 128
    static let melFrames       = 1_501            // fixed encoder mel width
    static let encoderChannels = 1_024
    static let decoderHidden   = 640
    static let decoderLayers   = 2                 // h/c are [2,1,640]

    // TDT decode
    static let vocabSize       = 1_024            // text tokens 0…1023
    static let blankId         = 1_024            // == vocabSize; not in vocab
    static let durationBins    = [0, 1, 2, 3, 4]  // num_extra_outputs = 5
    static let maxSymbolsPerStep = 10             // emissions cap per time index
}
```

---

## 3. Pipeline overview

```
audio file
  └─ decode + resample → Float32 mono @ 16 kHz                       (§4)
       └─ plan overlapping windows (center tiles audio, ±2.5 s ctx)  (§5)
            └─ per window:  preprocessor → encoder                   (§6)
                 └─ slice out the CENTER frames only                 (§6.3)
            └─ concatenate center slices → one [1,1024,ΣT] stream    (§6.4)
                 └─ ONE continuous TDT decode (decoder + joint)      (§7)
                      └─ token ids → text via vocab (▁ = space)      (§8)
                           └─ print transcript
```

Single continuous decode over the **concatenated center frames** is mandatory.
Decoding per-window and string-joining (the current Strategy-A harness) is the
fallback baseline only; per-slice decode of overlapping centers (Strategy D)
collapses (-20–30% words). See `parakeet-v2-longclip-decode-strategy` memory.

---

## 4. Audio loading (`AudioLoader`)

- Decode the file with `AVAudioFile` → `AVAudioPCMBuffer`.
- Convert to **mono Float32 @ 16 kHz** with `AVAudioConverter` (downmix multi-channel
  by averaging; resample any input rate). Output sample range expected in `[-1, 1]`.
- Return `[Float]` of length `n`. No additional normalization — the preprocessor
  performs NeMo's per-feature mel normalization internally (it is `dither=0`,
  `pad_to=0` in the exported graph; the Python harness feeds raw `[-1,1]` PCM).

Edge cases: empty/sub-window audio is valid (single window, §5).

---

## 5. Window planning (`WindowPlanner`) — pure, unit-testable

Direct port of `plan_windows` in `longform_transcribe.py`. The **center** regions
tile `[0, n)` contiguously and non-overlapping; each window additionally reads
`ctxSamples` of look-around that is encoded but **not** emitted.

```swift
struct WindowSpec {
    let winStart: Int   // first sample fed to preprocessor
    let trueLen:  Int   // real (unpadded) samples in the window  → audio_length
    let centerLo: Int   // first emitted sample (absolute)
    let centerHi: Int   // one-past-last emitted sample (absolute)
}

func planWindows(_ n: Int) -> [WindowSpec] {
    // Short audio: single full-context window, no seam (matches fast path).
    if n <= Const.winSamples {
        return [WindowSpec(winStart: 0, trueLen: n, centerLo: 0, centerHi: n)]
    }
    var specs: [WindowSpec] = []
    let center = Const.centerSamples
    var c0 = 0
    while c0 < n {
        let c1 = min(c0 + center, n)
        let winStart = max(0, c0 - Const.ctxSamples)        // clamp, never left-pad
        let trueLen  = min(Const.winSamples, n - winStart)
        specs.append(.init(winStart: winStart, trueLen: trueLen,
                           centerLo: c0, centerHi: c1))
        c0 += center
    }
    return specs
}
```

**Invariants (assert in tests):** centers are contiguous (`specs[i].centerHi ==
specs[i+1].centerLo`), cover `[0,n)`, never overlap; first window has no left context
(start of audio), last has no right context (end of audio) — exactly what NeMo sees at
true boundaries. Left padding is **never** used (it would corrupt tail-only length
masking); the window start clamps to 0 and the emit region shifts instead.

---

## 6. Encode each window (`Encoder`)

For each `WindowSpec`:

### 6.1 Feed the preprocessor
- Copy `wav[winStart ..< winStart + winSamples]` into a buffer; **right-zero-pad** to
  `winSamples` if the tail is short. (Preprocessor accepts flexible length, but feeding
  a fixed 240000 keeps the mel width deterministic; pass the *true* unpadded length.)
- `audio_signal` ← the (possibly padded) `[1, winSamples]` f32 buffer.
- `audio_length` ← `Int32(spec.trueLen)`.
- Run → `mel [1,128,T_mel]`, `mel_length`.

### 6.2 Feed the encoder
- Right-pad `mel` to `[1,128,1501]` (zeros). `mel_length` passes through unchanged.
- Run → `encoder [1,1024,188]`, `encoder_length`.
- `tValid = min(Int(encoder_length[0]), 188)`.

### 6.3 Extract the center-frame slice (`sliceForWindow`)
Port of `_slice_for_window`. Map the emitted **sample** region to encoder **frames**
using the window's own samples→frames ratio (robust to exact subsampling factor):

```swift
let ratio = Double(tValid) / Double(max(1, spec.trueLen))
var fs = Int((Double(spec.centerLo - spec.winStart) * ratio).rounded())
var fe = Int((Double(spec.centerHi - spec.winStart) * ratio).rounded())
fs = max(0, min(fs, tValid))
fe = max(fs, min(fe, tValid))
// emitted frames: encoder[:, :, fs ..< fe]   (drop if fe == fs)
```

### 6.4 Concatenate
Append each window's `[1,1024, fe-fs]` slice into one contiguous
`[1, 1024, ΣT]` Float32 buffer (channel-major, frame-minor — matches the encoder's
`[1,C,T]` layout). Track `ΣT` = total emitted frames. This buffer is the single
decode stream.

> **MLMultiArray stride caution.** ANE/CoreML outputs may be non-contiguous; read via
> the array's `strides`, not assumed row-major packing, when copying frames out (see
> `CoreMLParakeet.swift` `floats(...)` helper for the stride-aware read pattern). Build
> the concatenated buffer as a plain dense `[Float]` you own.

---

## 7. Continuous TDT decode (`TdtDecoder`) — the core

One pass over the `ΣT` concatenated frames. Semantics ported from
`TdtDecoderV3.swift`, simplified to a single non-streaming, non-batched stream.

### 7.1 State
```swift
var timeIdx   = 0                         // current encoder frame
var hidden    = zeros([2,1,640])          // LSTM h
var cell      = zeros([2,1,640])          // LSTM c
var lastToken = Const.blankId             // SOS = blank prime
var tokens:   [Int] = []                  // emitted text token ids (no blanks)
var emittedAtT = 0                        // emissions at current timeIdx
```

### 7.2 Prime the decoder (SOS)
Run `parakeet_decoder` once with `targets = [[blankId]]`, `target_length = [1]`,
zero `h_in`/`c_in` to get the initial `decoder_step [1,640,1]`, `h_out`, `c_out`.
Keep `decoderOut`, `hidden`, `cell`.

> NeMo's RNN-T predictor is primed with the blank id as start-of-sequence. The decoder
> embedding treats `blankId` as the SOS embedding; this matches the Python
> `decode_encoder` path (NeMo `rnnt_decoder_predictions_tensor`).

### 7.3 Main loop
```
while timeIdx < ΣT:
    encStep = encoder[:, :, timeIdx]               # [1,1024,1]
    (tok, dur) = joint_decision_single_step(encStep, decoderOut)   # argmax done in-model
    dur = mapDuration(dur)                          # durationBins[dur]; identity here

    if tok == blankId:
        # blank: advance time, do NOT update the decoder LSTM (silence carries
        # no language context). Force progress so we can't stall.
        if dur == 0 { dur = 1 }
        timeIdx   += dur
        emittedAtT = 0
    else:
        tokens.append(tok)
        # advance the predictor with the emitted token
        (decoderOut, hidden, cell) =
            decoder(targets: [[tok]], target_length: [1], h_in: hidden, c_in: cell)
        lastToken   = tok
        emittedAtT += 1
        timeIdx    += dur                           # TDT: token also skips `dur` frames

        # guard: cap emissions at one time index, force advance
        if emittedAtT >= maxSymbolsPerStep {
            timeIdx   += 1
            emittedAtT = 0
        }
```

**Key TDT rules (verified against `TdtDecoderV3.swift`):**
1. **Duration advances time** on every step — blank *and* non-blank. This is the TDT
   difference from plain RNN-T (where only blank advances time).
2. **Blank with `dur==0` is forced to `dur=1`** — otherwise the loop cannot make
   progress on a frame the model keeps labeling blank.
3. **Blank does not update the decoder LSTM** — only emitted tokens advance the
   predictor state. (Skipping decoder updates on blanks is what keeps language context
   continuous across silence.)
4. **`maxSymbolsPerStep` cap** prevents an infinite emit loop at one frame.
5. Because the decode is **continuous** over all `ΣT` frames, there is no per-window
   reset — `hidden`/`cell`/`lastToken` persist across former window boundaries. This is
   the load-bearing property that makes Strategy C work and Strategy D fail.

> No streaming finalization / token-stitcher is needed: this is offline, single-pass
> over the whole concatenated stream. The `TokenStitcher` overlap-merge in
> parakeet-unified is for *streaming* chunk deltas and is not used here (overlap is
> handled at the encoder-frame level, not the token level).

---

## 8. Detokenize (`ParakeetTokenizer`)

Reuse the exact logic of `ParakeetTokenizer.swift`:
- Load `parakeet_vocab.json` → `[Int: String]`.
- For each emitted id: skip empty pieces; if piece starts with `▁`, push a space then
  the remainder, else append the piece raw.
- Join, `trimmingCharacters(in: .whitespaces)`.

Blank ids never reach this stage (filtered in §7). Print the final string.

---

## 9. CoreML loading (`ModelRunner`)

- Load each `.mlmodelc` via `MLModel(contentsOf:configuration:)`.
- `MLModelConfiguration.computeUnits = .all` (encoder is ~99.96% ANE-resident INT8;
  preprocessor/decoder/joint fall back to CPU/GPU as the OS decides).
- Wrap each model in a small typed runner that builds `MLMultiArray`s of the exact
  shape/dtype in §1.1 and reads outputs stride-aware.
- **dtype:** all float I/O is `Float32` (`.float32` MLMultiArray) per the verified
  metadata — no fp16 conversion needed on the Swift side. ints are `.int32`.
- Reuse a single decoder/joint model instance across all steps (stateless calls;
  state lives in the `h/c` arrays we thread).

> Harmless teardown noise: `E5RT … slice_by_index: zero shape error` on process exit is
> a known CoreML teardown artifact, not a failure.

---

## 10. SwiftPM layout

```
parakeet-transcribe/
  Package.swift                      # executableTarget, macOS 14+ (.v14)
  Sources/parakeet-transcribe/
    main.swift                       # arg parsing, orchestration, print
    AudioLoader.swift                # AVFoundation decode/resample → [Float]
    WindowPlanner.swift              # planWindows (pure)
    Encoder.swift                    # preprocessor+encoder+sliceForWindow+concat
    TdtDecoder.swift                 # continuous TDT decode loop
    ParakeetTokenizer.swift          # copy from parakeet-unified
    ModelRunner.swift                # MLModel wrappers + MLMultiArray helpers
  Tests/.../WindowPlannerTests.swift # tiling invariants (§5)
```

CLI:
```
parakeet-transcribe --models <dir> --audio <file> [--ctx-s 2.5] [--baseline]
```
- `--models` dir containing the 4 `.mlmodelc` + `parakeet_vocab.json` (extract the
  vocab from the `.nemo` per §1.2 — do not assume one is already present).
- `--ctx-s` overrides context seconds each side (default 2.5; sets `ctxSamples`).
- `--baseline` (optional) runs Strategy A (per-window decode + join) for A/C comparison.
- Platform: macOS 14+ / iOS 17+ (matches the conversion target). Frameworks:
  `CoreML`, `AVFoundation`, `Accelerate` (optional, for any buffer math).

---

## 11. Validation

Parity target is `longform_transcribe.py` (the Python reference this ports). Acceptance:

1. **Unit:** `WindowPlannerTests` — centers contiguous, full coverage, no overlap, no
   left pad, for n ∈ {1s, 15s, 15.0001s, 31s, 66s}.
2. **Numeric parity (single window, ≤15s clip):** Swift transcript == Python
   `transcribe_longform` output token-for-token on a short clip (no overlap path
   exercised — isolates encoder/decoder/joint wiring).
3. **Long-clip parity (lsc, >15s):** Swift WER on lsc ≈ Python C = **0.0726**
   (tolerance for float-order differences; should match within a token or two per clip).
   Confirm Swift **beats** the per-window baseline (A = 0.0864) on the same clips.
4. **Sanity:** `--baseline` Swift output ≈ Python `transcribe_baseline` (Strategy A).

Reference numbers (lsc, 9 clips, INT8 `final` encoder):
`A 0.0864 · B 0.0932 · C(this) 0.0726 · D 0.2715 · NeMo 0.0667`.

---

## 12. Pitfalls (each cost real debugging in the Python work)

- **Plain-RNN-T decode loop is wrong for TDT.** Must read `duration` and advance
  `timeIdx` by it on every step. Ignoring duration desyncs the frame pointer.
- **Per-window / per-slice decode collapses** (Strategy D, 3× worse). Concatenate
  center frames first, then one decode.
- **Left-padding windows corrupts the mask** — clamp `winStart` to 0, shift the emit
  region; right-pad only.
- **Don't trust the static 188 frame count** — always slice with `encoder_length`.
- **Vocab ids ≥1024 are not text.** Extract the vocab from the `.nemo`'s
  `*_tokenizer.vocab` (1024 clean entries, §1.2). The FluidAudio dump agrees on 0…1023
  but appends rows 1024–1030 where `1024` (blank) maps to `"▁warming"` — a naïve lookup
  would emit "warming" for every blank. Filter blank before detokenizing regardless;
  only 0…1023 are real text tokens.
- **`blankId == vocabSize == 1024`**, durations are the **last 5** logits — but the
  single-step joint already argmaxes both, so Swift consumes `token_id`/`duration`
  directly and never touches raw logits.
- **MLMultiArray strides** — ANE outputs can be non-contiguous; copy stride-aware.
```
