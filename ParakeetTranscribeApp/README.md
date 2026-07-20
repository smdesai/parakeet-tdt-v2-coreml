# ParakeetTranscribe (iOS app)

A SwiftUI iOS app that runs **NVIDIA Parakeet-TDT-v2 (0.6B)** entirely on-device.
The app installs small and **downloads the four CoreML models (~620 MB) from
Hugging Face on first launch** ([`smdesai/parakeet-tdt-0.6b-v2-coreml`](https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml)),
showing a progress screen while it fetches them. After that first download the
app works **fully offline** — every subsequent launch detects the on-disk models
and goes straight to transcription. It reuses the `parakeet-transcribe` CLI's
verified Strategy-C core (encoder windowing + continuous TDT decode) and wraps it
in the dark "aurora" UI from `parakeet-unified/apps/ParakeetASR`.

Two ways to transcribe, both on-device:

| Mode | How | Decode |
|------|-----|--------|
| **File** | Folder button → pick any audio file | Batch Strategy C (one continuous decode over the whole clip) |
| **Mic**  | Mic button → record → stop | **Incremental** Strategy C — the sliding window runs *as you speak*, the transcript grows on screen |

## Build & run on a device

The models are **no longer staged into the app bundle** — there is no
`stage-models.sh` step before the build. The app fetches the models from Hugging
Face on first launch instead.

```bash
cd ParakeetTranscribeApp
xcodegen generate                # project.yml -> ParakeetTranscribe.xcodeproj
open ParakeetTranscribe.xcodeproj
# In Xcode: select your device, ensure signing team (TA92TEWDS4 / your own), Run.
```

On **first launch** the app downloads the four CoreML models + vocab (~620 MB)
from [`smdesai/parakeet-tdt-0.6b-v2-coreml`](https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml)
into Application Support, behind a progress screen (determinate bar, "file X of
N", MB counter, current file name). The app **requires a network connection on
first run**; if the download is interrupted it resumes on the next launch (files
already fully written are skipped by size). Every launch after the models are
installed works **fully offline** and skips straight to model load.

Requirements: Xcode 27+/iOS 17+ device (deployment target iOS 27.0, matching the
reference app), a network connection on first run, and **~620 MB free** for the
downloaded models (stored in Application Support, not bundled). The `.ipa` itself
is now small — it no longer carries the model payload.

### Compile-check (no device)

```bash
xcodebuild -project ParakeetTranscribe.xcodeproj -scheme ParakeetTranscribe \
  -sdk iphoneos -configuration Debug -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

### Tests

Pure-logic unit tests for the model-download planning seams live in `Tests/`
(target `ParakeetTranscribeTests`, wired into the app scheme). They exercise the
allowlist filter, resolve-URL construction, resume-by-size, and total-bytes
accounting with a captured HF tree fixture — no CoreML, no live network. Run them
from Xcode (Cmd-U) or:

```bash
xcodebuild test -project ParakeetTranscribe.xcodeproj -scheme ParakeetTranscribe \
  -destination 'platform=iOS Simulator,name=iPhone 16'
```

## How the microphone uses the sliding window

Strategy C is a **batch/offline** algorithm — it tiles the audio with overlapping
15 s windows (each emits a 10 s center and reads 2.5 s of look-around on both
sides) and decodes the concatenated center frames in one continuous TDT pass.

The mic path does the *same* work **progressively** (`StreamingTranscriber`):

1. The recorder yields 16 kHz mono chunks as they arrive (`AsyncStream`).
2. A window is **finalized** the instant its full 15 s span — center **plus its
   2.5 s right-context** — has been recorded. Encoded then, its output is identical
   to the offline run's (the right-context is real audio, never zero-padded).
3. Each finalized window's center frames are appended to a growing stream and a
   **single continuous TDT decode** advances over the new frames, carrying the
   LSTM/emission state forward. No frame is decoded twice or early.
4. On stop, the trailing partial window is flushed (no right-context — same as the
   batch path's last window), giving the final transcript.

**Caveat (honest):** this is *progressive*, not word-by-word real-time. Because a
window needs ~2.5 s of future audio before it's faithful and the 568 MB encoder
runs per window, the transcript lags the speaker by ≈2.5 s + encode time and
updates in ~10 s chunks. We deliberately do **not** zero-pad the missing
right-context to update faster, because that would diverge from the validated
Strategy-C accuracy. The on-device result matches the CLI.

## Source map

```
project.yml                         XcodeGen config (iOS app + ParakeetTranscribeTests unit-test target)
scripts/stage-models.sh             (no longer part of the build) populates the HF source repo, not the bundle
Resources/Assets.xcassets/          accent + launch colors, app icon slot
Sources/
  App/ParakeetTranscribeApp.swift   @main; injects TranscriptionEngine; dark theme
  Theme/Theme.swift                 aurora palette + glassCard (from ParakeetASR)
  Views/RootView.swift              background + download / progress / failure overlays
  Views/TranscriptionView.swift     header, transcript card, RTFx metrics, mic + file buttons
  Views/Components/WaveformView.swift  rolling level meter
  Engine/TranscriptionEngine.swift  ObservableObject; downloads then loads models; file + mic flows
  Engine/ModelDownloader.swift      first-launch HF download (tree API + resolve URLs, resume, sentinel)
  Engine/MicRecorder.swift          AVAudioEngine capture -> AsyncStream<[Float]> @ 16 kHz + level
  Engine/AudioFileSamples.swift     security-scoped file URL -> mono 16 kHz [Float]
  Engine/StreamingTranscriber.swift incremental Strategy C (sliding window as audio arrives)
  Core/                             VERBATIM copies of the CLI core:
    Const, MLArray, ModelRunner, Encoder, TdtDecoder,
    WindowPlanner, ParakeetTokenizer, Transcriber
Tests/
  ModelDownloaderTests.swift        pure-logic tests for the downloader's planning seams
```

The `Core/` files are unmodified copies of
`../parakeet-transcribe/Sources/parakeet-transcribe/` (minus `main.swift` and the
path-based `AudioLoader.swift`, which are CLI-only). Keeping them byte-identical
means the app inherits the CLI's verified behavior — including the
empty-output → CPU-retry fix for the rare ANE onset-collapse
(`../parakeet-transcribe/docs/ane-onset-collapse.md`).

## Models

Downloaded on first launch from
[`smdesai/parakeet-tdt-0.6b-v2-coreml`](https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml)
(the INT8-encoder pipeline), into Application Support (~620 MB, backup-excluded):

| Model | Size | Role |
|-------|------|------|
| `parakeet_preprocessor.mlmodelc` | 620 KB | audio → mel `[1,128,1501]` |
| `parakeet_encoder.mlmodelc` | 568 MB | mel → encoder `[1,1024,188]` (FastConformer) |
| `parakeet_decoder.mlmodelc` | 14 MB | LSTM predictor |
| `parakeet_joint_decision_single_step.mlmodelc` | 3.3 MB | token + duration argmax |
| `parakeet_vocab.json` | 16 KB | 1024-entry SentencePiece vocab |

All run on `.all` compute units (ANE-first); the file path retries on CPU if a
clip collapses to empty (see the CLI docs).
```
