# Parakeet-TDT-v2 (0.6B) CoreML — Transcriber Throughput Optimizations

A running account of the performance work on the **Swift long-form transcriber**
(Strategy C), covering both the CLI (`parakeet-transcribe/`) and the iOS app
(`ParakeetTranscribeApp/`). Every step preserved **byte-identical** transcripts —
the accuracy gate was an exact-match SHA, never WER.

**Headline result:** RTFx **153.76x → 340x** on Mac (66-min clip) and
**157x → 245x** on a real iOS device (97-min clip), with the decoded token
sequence unchanged at every step.

---

## 1. The pipeline being optimized

Strategy C transcribes long audio with **overlapping 15 s windows** (center 10 s +
2.5 s context each side) and ONE continuous TDT decode over the concatenated
center frames. It drives four CoreML models, each pinned to a different engine:

| Model | Role | Engine | Cost profile |
|---|---|---|---|
| `parakeet_preprocessor` | audio → mel | CPU + GPU | ~per-window, moderate |
| `parakeet_encoder` (INT8, 568 MB) | mel → `[1,1024,188]` | **ANE (100%)** | dominant, ~27 ms/window |
| `parakeet_decoder` | LSTM predictor step | CPU | per emitted token |
| `parakeet_joint_decision_single_step` | token + duration argmax | CPU | per frame |

The encoder is the single most expensive stage and runs entirely on the ANE; the
preprocessor (CPU+GPU) and the decode loop (decoder + joint, both CPU) run on
different engines. That separation is what makes pipelining possible — different
engines can run concurrently.

---

## 2. Optimization timeline

### Step 0 — Baseline (sequential): 153.76x

The original Strategy C ran in **two serial phases on two different engines**:

1. **Encode phase** — for every window: preprocessor → encoder, collecting all
   center-frame slices into one big stream. (~17 s on the 66-min clip: preproc
   5.6 s + encoder 11.4 s.)
2. **Decode phase** — one continuous TDT decode over the whole stream.
   (~7.7 s: decoder 4.5 s + joint 3.2 s, both CPU.)

Each engine sat **idle for roughly half the run** — the ANE was idle during the
entire decode phase, and the CPU was idle during the entire encode phase.

---

### Step 1 — Two-stage pipeline (encode ∥ decode): ~207x

**Idea:** overlap the two phases instead of running them back-to-back.

A producer `Thread` runs the encode loop and hands each finished center slice to a
bounded single-producer/single-consumer queue; the main thread runs a **resumable**
TDT decoder (`StreamingTdtDecoder`) that consumes slices as they arrive. The decode
now hides under the encode.

The key enabler is that TDT decode is a **resumable state machine** (LSTM state +
per-frame emission counting + force-blank guard). When a duration skip lands past
the frames decoded so far, it simply pauses and resumes when more arrive — so
incremental decode produces the **exact same tokens** as one-shot decode. Only the
*scheduling* differs.

- Wall: ~25.8 s → ~19.3 s. **RTFx 153.76x → ~207x.**
- Commit `d54683e` (2-stage at 206.9x), later subsumed by the 3-stage work.
- Output byte-identical (clean SHA `e93b6b90`, 12,507 words).

---

### Step 2 — Three-stage pipeline (preprocess ∥ encode ∥ decode): 312x (Mac)

**Observation:** the producer thread in Step 1 still ran preprocessor (CPU+GPU,
~15 ms) → encoder (ANE, ~31 ms) **serially per window**, so the ANE idled during
every single preprocess.

**Fix:** split into a true **3-stage pipeline**, one engine per stage:

```
stage 1: preprocess (CPU+GPU)  ──melQueue──▶  stage 2: encode (ANE)  ──sliceQueue──▶  stage 3: decode (CPU)
   preThread                                      encThread                              main thread
```

- `Encoder.encodeStreaming` was factored into `preprocessWindow` (stage 1) and
  `encodeWindow` (stage 2).
- Stages are chained by two `BoundedQueue<T>` instances — a generic NSCondition
  bounded blocking queue, single-producer/single-consumer, with amortized-O(1)
  dequeue (head index + occasional compaction). `push` blocks when full so the
  producer can't run unboundedly ahead (caps memory).
- The encoder now runs **back-to-back**; the per-window preprocess hides under it.

The wall-clock floor is now ≈ the encoder's pure ANE time (everything else hides
under it).

- Wall: 19.4 s → 12.8 s. **RTFx ~207x → 312x** (stable 311–312x over 3 runs).
- Commit `dae564c`. Output byte-identical (12,507 words).
- Exceeded the stated goal (≥220x) comfortably.

Files: `Encoder.swift`, `Transcriber.swift` (`decodeStrategyC` rewritten as the
3-stage pipeline + `BoundedQueue<T>`), `TdtDecoder.swift` (`StreamingTdtDecoder`).

---

### Step 3 — Move the slice transpose off the contended decode thread: Mac 318x→340x, **device 188x→245x**

This step came from **on-device profiling**, which revealed the pipeline behaved
very differently on a phone than on the Mac.

#### The diagnosis

Per-stage profiling was added to the app (see §3) and measured on a real device
(97:19 audio):

| stage | engine | busy |
|---|---|---|
| encoder | ANE | 24.2 s ← the hard floor |
| decoder | CPU | 11.5 s |
| joint | CPU | 6.19 s |
| preprocessor | CPU+GPU | 10.0 s |
| **Σ busy** | | **51.9 s** |
| **wall** | | **31.1 s** → 188x |
| **overlap** | | **75%** |

The pipeline *was* overlapping (75% — Σ busy 51.9 s compressed to 31.1 s wall), but
wall was **6.9 s above the 24.2 s ANE encoder floor**. On the Mac (many cores) the
non-ANE CPU work hid completely and wall ≈ encoder; on the phone (few performance
cores) the combined CPU work (decode 11.5 + joint 6.19 + preproc 10.0 = 27.7 s)
**rivaled the ANE budget and couldn't fully hide**, so the ANE stalled waiting to
be fed/drained. The device was **CPU-bound, not ANE-bound**.

#### The root inefficiency

Each window's encoder output was copied **twice**, and the second copy ran on the
contended decode thread:

- `Encoder.extractSlice` (encode thread) emitted a **C-major** slice
  (`data[c*len + t]`).
- `StreamingTdtDecoder.append` (decode thread) then **transposed** it to
  **frame-major** (`frames[t*channels + c]`) so each per-frame channel vector was
  contiguous for `fillStep`. That transpose was ~65 M element copies per run,
  burning cycles on the exact thread that was the bottleneck.

#### The fix

Make `extractSlice` produce **frame-major `[len, channels]` slices directly**, so:

- The transpose now runs on the **encode thread**, which otherwise idles waiting on
  the ANE.
- `StreamingTdtDecoder.append` becomes a flat bulk copy
  (`frames.append(contentsOf: slice.data)`) — no per-element work on the decode
  thread.
- `Encoder.concat` (used only by the non-hot **baseline** `encode()` path)
  transposes frame-major back to C-major, so `EncodedStream` / `TdtDecoder` /
  `fillEncoderStep` are completely untouched.

This is a **pure index reshuffle — zero float arithmetic changed** — so the decode
input is bit-identical by construction.

#### Results

**Device (97:19 audio):**

| metric | before | after |
|---|---|---|
| RTFx | 188x | **245x** |
| wall | 31.1 s | **23.8 s** |
| overlap | 75% | **94%** |
| Σ busy | 51.9 s | **46.3 s** |
| encoder | 24.2 s | 22.4 s |
| decoder | 11.5 s | **9.24 s** |
| joint | 6.19 s | 5.29 s |
| preprocessor | 10.0 s | 9.35 s |

Two effects, both confirming the diagnosis:

1. **Overlap 75% → 94%** — the full ~6.9 s of recoverable headroom was reclaimed.
   Wall (23.8 s) is now only 1.4 s above the 22.4 s ANE floor.
2. **Σ busy itself dropped 51.9 s → 46.3 s** (decoder 11.5 → 9.24 s) — a bonus.
   Freeing the decode thread of the transpose reduced contention on *every*
   concurrent CoreML prediction, not just unblocking the pipeline. This is why the
   result (245x) beat the conservative 200–215x estimate.

**Mac (66-min clip):** ~318x → **~340x** (transcribe 12.5 s → 11.7 s). Smaller
relative gain because the Mac was already core-rich and near its floor.

- CLI commit `c7cb1c0`.
- Files: `Encoder.swift` (`extractSlice` + `concat`), `TdtDecoder.swift` (`append`).

#### Practical ceiling

Device wall is now 1.4 s above the 22.4 s ANE encoder floor (absolute ceiling
≈ 260x). The remaining 1.4 s is pipeline fill/drain tail and isn't worth chasing.
The 17.7 s sequential TDT decode is inherent (can't be parallelized) but is now
fully hidden under the encode.

---

## 3. Supporting infrastructure

### Per-stage profiling (`Profile`)

A lightweight per-model wall-time accumulator (`times`/`counts` dicts under an
`NSLock`) in `ModelRunner.swift`. `predict()` records each call against a label
derived from model identity (`===`).

- **CLI:** gated behind `PARAKEET_PROFILE=1`.
- **App:** always-on (environment variables don't reach an untethered device);
  `reset()` before each run, `snapshot()` after.

The **overlap metric** is the key diagnostic:

```
overlap% = (Σbusy − wall) / (Σbusy − maxStage)
```

- 0% = fully serialized (wall = sum of stages)
- 100% = fully overlapped (wall = the single busiest stage, i.e. the ANE floor)

### App metrics UI

`TranscriptionView.swift` (`BatchPerformanceView`) shows the RTFx headline always,
with the per-stage breakdown (time / % per stage + the `Σ busy · wall · overlap`
verdict line) **collapsed behind a tap-to-expand dropdown** (chevron, default
collapsed). The chevron and tap target only render when there are stages to show
(the file path); the mic path has no per-stage data, so its header stays a plain
row.

### `--no-transcript` flag

`main.swift` gained a `--no-transcript` flag and an always-on RTFx/timing line on
**stderr**, so stdout stays clean for WER pipelines while still reporting speed.

---

## 4. How byte-identical parity was proven

The accuracy gate is **exact-match**, not WER. The verification method that worked:

```bash
# Build the pre-edit (committed) version in a scratch dir — working tree untouched.
git archive HEAD --prefix=cli_before_src/ | (cd /tmp && tar -xf -)
cd /tmp/cli_before_src && swift build -c release

# Run before and after on the SAME machine, SAME audio/models, then cmp.
```

**Critical gotcha — CoreML stdout noise.** CoreML emits a non-deterministic C++ log
onto **stdout**, glued to the transcript start:

```
E5RT encountered an STL exception. msg = Failed to PropagateInputTensorShapes:
std::runtime_error during type inference for ios17.slice_by_index: zero shape error.
```

It's harmless (~19 words) but it breaks naive SHA/word-count comparison. **Strip it
before hashing:**

```bash
... 2>/dev/null | sed -E 's/^.*zero shape error\.//' | shasum
```

- **Accuracy gate (clean):** SHA `e93b6b90…`, **12,507 words**.
- Same-machine before/after `diff` and `cmp` of the cleaned transcripts:
  **byte-for-byte identical** at every optimization step.
- Note: an earlier capture reported SHA `2d904410` — that was computed under
  different normalization (trailing newline / strip method). The trustworthy gate
  is the cleaned `e93b6b90` content hash, confirmed via same-machine comparison.

The RTFx report goes to **stderr**, not stdout — don't discard stderr when
measuring speed (`2>&1 1>/dev/null`).

---

## 5. Why each change is accuracy-safe (the general argument)

Every optimization here is **scheduling- or layout-only**:

- Same windows, same mel features, same encoder frames, same emit regions.
- Same TDT decode state machine (same priming, same per-frame emission counting,
  same force-blank guard, same duration mapping).
- The pipeline changes only *when* work runs (concurrency); the transpose change
  only *where* values sit in memory (index order). No float arithmetic is altered.

There is also a separate, pre-existing correctness guard unrelated to speed: a rare
ANE/GPU fp16 "onset collapse" (~1 in 1400 clips) where soft speech after leading
silence decodes to empty; `Transcriber.transcribe` detects the empty result and
retries once on a CPU fp32 runner (commit `436caf2`). This is preserved by all
pipeline changes.

---

## 6. Summary table

| Step | Change | Mac RTFx | Device RTFx | Commit |
|---|---|---|---|---|
| 0 | Sequential Strategy C | 153.76x | — | — |
| 1 | 2-stage (encode ∥ decode) | ~207x | — | `d54683e` |
| 2 | 3-stage (+ preprocess, `BoundedQueue`) | **312x** | 188x | `dae564c` |
| 3 | Frame-major slices (transpose → encode thread) | **340x** | **245x** | `c7cb1c0` (CLI) |

All steps: output byte-identical (clean SHA `e93b6b90`, 12,507 words).

### Status

- **CLI:** committed locally (`c7cb1c0`, `dae564c`); not pushed.
- **App (`ParakeetTranscribeApp/`):** Step 3 + profiling + metrics-dropdown ported;
  builds for arm64 iOS; 245x confirmed on device. Currently **untracked**
  (bundles ~585 MB of CoreML models — needs a `.gitignore` before any commit).

### Ideas not pursued (all accuracy-safe, if more speed is ever needed)

- Reuse `MLDictionaryFeatureProvider` across the ~50 k predict calls (~1.5 s).
- Narrow `MLArray.floats` to only the emitted encoder frames instead of the full
  `[1024,188]` read.
- Preprocessor CPU → GPU (unverified; conversion pins it CPU+GPU for ANE dim
  limits).

Rejected outright: a lower-bit encoder (accuracy cliff) and smaller windows (less
context, accuracy loss).
