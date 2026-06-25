# ParakeetTranscribe (iOS app)

A SwiftUI iOS app that runs **NVIDIA Parakeet-TDT-v2 (0.6B)** entirely on-device,
with the four CoreML models **bundled as app resources** (no runtime download).
It reuses the `parakeet-transcribe` CLI's verified Strategy-C core (encoder windowing
+ continuous TDT decode) and wraps it in the dark "aurora" UI from
`parakeet-unified/apps/ParakeetASR`.

Two ways to transcribe, both on-device:

| Mode | How | Decode |
|------|-----|--------|
| **File** | Folder button → pick any audio file | Batch Strategy C (one continuous decode over the whole clip) |
| **Mic**  | Mic button → record → stop | **Incremental** Strategy C — the sliding window runs *as you speak*, the transcript grows on screen |

## Build & run on a device

```bash
cd ParakeetTranscribeApp
./scripts/stage-models.sh        # copy ../parakeet_coreml_v2_final/*.mlmodelc + vocab into Resources/Models
xcodegen generate                # project.yml -> ParakeetTranscribe.xcodeproj
open ParakeetTranscribe.xcodeproj
# In Xcode: select your device, ensure signing team (TA92TEWDS4 / your own), Run.
```

Requirements: Xcode 27+/iOS 17+ device (deployment target iOS 27.0, matching the
reference app), ~590 MB free for the embedded models.

### Compile-check (no device)

```bash
xcodebuild -project ParakeetTranscribe.xcodeproj -scheme ParakeetTranscribe \
  -sdk iphoneos -configuration Debug -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
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
project.yml                         XcodeGen config (iOS app, models as a folder reference)
scripts/stage-models.sh             copy ../parakeet_coreml_v2_final -> Resources/Models
Resources/Models/                   bundled .mlmodelc + parakeet_vocab.json (staged; ~585 MB)
Resources/Assets.xcassets/          accent + launch colors, app icon slot
Sources/
  App/ParakeetTranscribeApp.swift   @main; injects TranscriptionEngine; dark theme
  Theme/Theme.swift                 aurora palette + glassCard (from ParakeetASR)
  Views/RootView.swift              background + progress / failure overlays
  Views/TranscriptionView.swift     header, transcript card, RTFx metrics, mic + file buttons
  Views/Components/WaveformView.swift  rolling level meter
  Engine/TranscriptionEngine.swift  ObservableObject; loads bundled models; file + mic flows
  Engine/MicRecorder.swift          AVAudioEngine capture -> AsyncStream<[Float]> @ 16 kHz + level
  Engine/AudioFileSamples.swift     security-scoped file URL -> mono 16 kHz [Float]
  Engine/StreamingTranscriber.swift incremental Strategy C (sliding window as audio arrives)
  Core/                             VERBATIM copies of the CLI core:
    Const, MLArray, ModelRunner, Encoder, TdtDecoder,
    WindowPlanner, ParakeetTokenizer, Transcriber
```

The `Core/` files are unmodified copies of
`../parakeet-transcribe/Sources/parakeet-transcribe/` (minus `main.swift` and the
path-based `AudioLoader.swift`, which are CLI-only). Keeping them byte-identical
means the app inherits the CLI's verified behavior — including the
empty-output → CPU-retry fix for the rare ANE onset-collapse
(`../parakeet-transcribe/docs/ane-onset-collapse.md`).

## Models

Bundled from `../parakeet_coreml_v2_final` (the INT8-encoder pipeline):

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
