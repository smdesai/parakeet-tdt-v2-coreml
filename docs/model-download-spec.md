# Spec — Download Parakeet CoreML models on first launch (iOS app)

**Status:** ready-for-agent
**Area:** `ParakeetTranscribeApp` (SwiftUI iOS app)
**Related:** `docs/swift-port-spec.md` (the Strategy-C core this app wraps)

---

## Problem Statement

The `ParakeetTranscribe` iOS app currently ships the four Parakeet-TDT-v2 CoreML
models **bundled as app resources** (`Resources/Models`, staged by
`scripts/stage-models.sh`, added to the target as a `type: folder` reference). The
INT8 encoder weight alone is ~594 MB, so the `.ipa` is well over half a gigabyte
before any app code. From the user's perspective this means:

- A very large download from the App Store / TestFlight before the app will even
  open, on a cellular or slow connection, with no in-app feedback about what the
  size is for.
- Every app update re-ships the full model payload even when only app code changed.
- The models are baked into the signed bundle, so there is no way to refresh them
  (e.g. a re-quantized encoder) without a full app resubmission.

The user wants the app to be small to install, and to fetch the model bundle **once,
on first launch, over the network**, with a clear progress bar — the same
first-run experience as the MagpieTTS app.

## Solution

On first launch the app has **no bundled models**. Instead it downloads the model
bundle from the public Hugging Face repo
[`smdesai/parakeet-tdt-0.6b-v2-coreml`](https://huggingface.co/smdesai/parakeet-tdt-0.6b-v2-coreml/tree/main)
into a writable, backup-excluded directory under Application Support, showing a
first-run progress screen with a determinate progress bar, a "file X of N" counter,
a byte/MB readout, and the name of the file currently downloading. When the download
finishes, the app proceeds straight into the existing "Loading Parakeet models…"
compile/map step and then to the transcription UI.

On every subsequent launch the app detects the completed on-disk bundle (via a
`.complete` sentinel) and skips the download entirely, going directly to model load.
If a previous download was interrupted (app killed, connectivity lost), the next
launch resumes it — files already fully written are skipped by size, so only the
remainder is fetched. Network/HTTP failures surface in the existing failure overlay
with a **Retry** button.

From the user's perspective: the app installs fast, downloads the model once with
honest progress, works fully offline forever after, and never silently blocks on a
half-finished download.

## User Stories

1. As a new user, I want the app to install quickly from the App Store, so that I'm not blocked by a >600 MB download before I can try it.
2. As a first-time user, I want the app to fetch its speech models automatically on first launch, so that I don't have to find or sideload model files myself.
3. As a first-time user, I want a progress bar while the models download, so that I know the app is working and roughly how long it will take.
4. As a first-time user, I want to see how much has downloaded (MB completed of total, and file X of N), so that the wait feels finite and understandable.
5. As a first-time user, I want to see the name of the file currently downloading, so that I can tell progress is real and not stalled.
6. As a returning user, I want the app to skip the download on the second launch, so that I get to transcribing immediately without re-downloading ~620 MB.
7. As a user on a slow connection, I want the download to keep going even if throughput dips, so that a temporary slowdown doesn't fail the whole install.
8. As a user whose app was killed mid-download, I want the next launch to resume where it left off, so that I don't re-download the parts I already have.
9. As a user who lost connectivity mid-download, I want a clear error and a Retry button, so that I can restart the download once I'm back online without reinstalling the app.
10. As an offline user who has already downloaded the models once, I want transcription to work with no network at all, so that the app remains a fully on-device tool after setup.
11. As a privacy-conscious user, I want only the model artifacts to be fetched from Hugging Face and nothing about my audio to leave the device, so that on-device transcription's privacy guarantee is preserved.
12. As a user, I want the downloaded models excluded from iCloud/device backup, so that ~620 MB of re-downloadable weights don't bloat my backups.
13. As a user with limited storage, I want the app to not keep a second bundled copy of the models inside the app, so that I only pay the model's disk cost once.
14. As a user, I want a partial/corrupt download to be treated as "not installed" on the next launch, so that I never end up stuck trying to load an incomplete model set.
15. As a user, I want the download to fetch only the four runtime models plus the vocabulary the app actually uses, so that I don't wait on artifacts the app never loads.
16. As a developer, I want the model source repo and revision to be defined in one place, so that pointing the app at a new model export is a one-line change.
17. As a developer, I want the download-planning logic (which files, where they go, what URL) to be unit-testable without hitting the network, so that I can verify correctness deterministically in CI.
18. As a developer, I want the existing Strategy-C transcription core (`ModelRunner`, `Transcriber`, `StreamingTranscriber`, `ParakeetTokenizer`) to be unchanged, so that the verified decode path keeps its byte-for-byte behavior.
19. As a developer, I want the download step to slot in ahead of the existing model-load step behind the same `prepareIfNeeded()` entry point, so that the rest of the app (file transcription, mic streaming, metrics) needs no changes.
20. As a developer, I want the app to fail loudly with an actionable message if the remote repo returns no files, so that a misconfigured repo id is caught immediately.
21. As a returning user, I want the "downloading" screen to appear only when a download is actually needed, so that my normal launches aren't interrupted by a spurious progress screen.
22. As a user, I want the app to show the download progress as a distinct phase from the "compiling/mapping models" phase, so that the two waits are not conflated into one ambiguous spinner.
23. As a user re-running after a failed download, I want Retry to resume rather than restart from zero, so that a failure near the end doesn't cost me the whole download again.

## Implementation Decisions

### Model source & on-device layout

- **Source of truth:** Hugging Face repo `smdesai/parakeet-tdt-0.6b-v2-coreml`,
  revision `main`. Repo id + revision live in a single `HFRepo` config enum (ported
  from the reference `ModelDownloader`), so re-pointing the app is a one-line change.
- Files are enumerated via the HF **tree API**
  (`https://huggingface.co/api/models/<id>/tree/<rev>?recursive=true`) and fetched via
  the **resolve** URL (`https://huggingface.co/<id>/resolve/<rev>/<path>`).
- **On-device install root:** a writable directory under Application Support (e.g.
  `Application Support/parakeet-tdt-v2-coreml/`), created on demand and marked
  `isExcludedFromBackup = true`. The repo tree is flat — the four `.mlmodelc`
  directories and `parakeet_vocab.json` sit directly under the root — so the install
  root **is** the `modelsDir` the rest of the app already consumes.
- **Install sentinel:** a `.complete` empty file written **last**, only after every
  file has been successfully written. `isInstalled()` checks solely for this sentinel;
  a partial install (no sentinel) is treated as "not installed" and re-walked on next
  launch (per-file size checks make this an effective resume, not a restart).

### What to download (and what to skip)

The repo has been trimmed to the runtime set (the raw-logit
`parakeet_joint_logits_single_step.mlmodelc` was removed since the app never loads
it). The downloader still filters the tree defensively so non-runtime files aren't
fetched:

- **Download:** `parakeet_preprocessor.mlmodelc`, `parakeet_encoder.mlmodelc`,
  `parakeet_decoder.mlmodelc`, `parakeet_joint_decision_single_step.mlmodelc` (all
  their inner files: `coremldata.bin`, `metadata.json`, `model.mil`,
  `analytics/coremldata.bin`, `weights/weight.bin`), plus `parakeet_vocab.json`.
- **Skip:** `.gitattributes`, and any hidden (`.`-prefixed) or `README.md` files.
- Total download after filtering ≈ **~613 MB** (encoder weight ~594 MB dominates).
- Filtering is expressed as an **allowlist of top-level model directory names + the
  vocab filename**, not a denylist, so a future stray artifact re-added to the repo
  (e.g. the logits joint) is excluded by default rather than shipped by accident.

### Downloader module (ported from MagpieTTS `ModelDownloader.swift`)

- Add a `ModelDownloader` type to the app (new file under `Sources/`, e.g.
  `Sources/Engine/ModelDownloader.swift`), adapted from
  `/Users/sdesai/Tools/MLX/Magpie/Sources/MagpieTTS/ModelDownloader.swift`. Reused
  wholesale: tree-API listing, `resolve`-URL construction, sentinel-gated
  `isInstalled()`, resume-by-size, streamed chunked writes to a `.tmp` then atomic
  rename, `DownloadProgress` snapshot struct, and `ModelDownloadError`.
- **Adapted for this app:**
  - `HFRepo.id` → `smdesai/parakeet-tdt-0.6b-v2-coreml`.
  - Install root folder name → `parakeet-tdt-v2-coreml`.
  - Runtime-set allowlist filter added to `fetchFileList()` (see above) in place of
    Magpie's tokenizer/speaker-embedding shape.
- **Progress contract** (unchanged from reference): `DownloadProgress { bytesCompleted,
  bytesTotal, currentFileIndex, totalFiles, currentFileName }`, emitted on the main
  actor. `bytesTotal` is the sum of tree-API sizes (LFS entries report resolved size,
  so it's accurate).
- **Public API:** `ensureInstalled(onProgress:) async throws` — returns immediately if
  the sentinel exists, otherwise downloads and reports progress; `static
  rootDirectory() -> URL` and `static isInstalled() -> Bool`.

### Engine integration (the single production seam)

- The only production change to the pipeline is at the model-directory seam.
  Today `TranscriptionEngine.bundledModelsDir()` returns `Bundle.main/Models`. It is
  replaced by `ModelDownloader.rootDirectory()` — the downloaded install root — and
  the "bundled models not found" precondition becomes the sentinel/`isInstalled`
  check. Everything downstream (`ModelRunner`, `ParakeetTokenizer`, `Transcriber`,
  `StreamingTranscriber`) already takes `modelsDir: URL` and is untouched.
- `prepareIfNeeded()` gains a **download step before load**: if not installed, call
  `downloader.ensureInstalled { progress in … }` and publish progress; on success,
  fall through to the existing `loadModels()`. The download runs before models are
  loaded, so there is no ordering hazard with the worker queue.
- **Phase model:** add a `downloading(DownloadProgress)` case to
  `TranscriptionEngine.Phase` (a new state distinct from `.preparing`), carrying the
  latest progress snapshot for the view. `prepareIfNeeded()` transitions
  `idle → downloading(…) → preparing → ready`, or `→ failed(String)` on any error.
  Progress updates re-publish `downloading(newSnapshot)`. This keeps download and
  "compile/map" as visibly separate waits.

  Prototype — the phase/state shape this encodes:
  ```swift
  enum Phase: Equatable {
      case idle
      case downloading(DownloadProgress)   // NEW — first-run model fetch
      case preparing                       // existing: compile + map weights
      case ready
      case transcribing
      case recording
      case failed(String)
  }
  ```

### UI (progress screen)

- `RootView` gains a branch: when `engine.phase` is `.downloading(let p)`, show a
  **determinate** progress screen instead of the indeterminate "Loading Parakeet
  models…" overlay. It renders:
  - a `ProgressView(value: fraction)` where `fraction = Double(bytesCompleted) /
    Double(max(bytesTotal, 1))`,
  - a byte readout ("312 MB of 613 MB", `ByteCountFormatter`),
  - "File \(currentFileIndex) of \(totalFiles)" and `currentFileName`.
- The existing indeterminate overlay is retained for `.preparing` (compile/map) and
  `.transcribing`. The existing `FailureOverlay` + **Retry** (`prepareIfNeeded()`) is
  reused verbatim — Retry naturally resumes because `ensureInstalled` skips
  already-downloaded files.

### Build config (remove the bundled model payload)

- Remove the `Resources/Models` `type: folder` resource from `project.yml` so the
  models are no longer copied into the app bundle. Update the target's stated size
  requirement.
- `scripts/stage-models.sh` is no longer part of the build flow; keep it (or a note)
  as the tool that populates the **HF repo** source, not the app bundle. `README.md`
  build steps updated: no `stage-models.sh` before build; note first-launch download
  and ~620 MB free-space / network requirement.
- Add `NSAppTransportSecurity` considerations only if needed — `huggingface.co` is
  HTTPS, so default ATS suffices; no Info.plist ATS exception.

## Testing Decisions

**What makes a good test here:** exercise only externally observable behavior of the
downloader's **pure, deterministic seams** — given a fixed HF-tree JSON payload and a
temp directory, assert the *download plan* and the *install-state logic*, never the
private buffering/threading internals. No live network in tests: the network call
(`fetchFileList`) and the byte-streaming (`downloadFile`) are the impure edges and are
out of scope for unit tests; the logic between them is what we test. This mirrors
`WindowPlannerTests` in the CLI, which tests the pure windowing math without touching
CoreML.

**New test target:** add `ParakeetTranscribeTests` (the app currently has
`testTargets: []`). Tests target the download-planning logic, which should be factored
so it is callable without a URLSession:

- **Tree → plan:** feed a captured tree-API JSON fixture (the real
  `smdesai/parakeet-tdt-0.6b-v2-coreml` tree) and assert the resulting file list:
  the four runtime `.mlmodelc` trees + `parakeet_vocab.json` are included;
  `.gitattributes`, hidden files, and `README.md` are excluded; entries are sorted
  deterministically. Include a synthetic non-runtime artifact (e.g. a stray
  `*_logits_*.mlmodelc` entry) in the fixture to prove the allowlist excludes it.
- **Resolve URL construction:** assert each relative path maps to the correct
  `https://huggingface.co/<id>/resolve/main/<path>` URL, including nested paths like
  `parakeet_encoder.mlmodelc/weights/weight.bin`.
- **Sentinel / `isInstalled`:** against a temp root, assert `isInstalled()` is false
  with no sentinel, true after the sentinel is written, and false again if the
  sentinel is absent even when other files exist (partial install ⇒ not installed).
- **Resume-by-size:** given a temp root pre-seeded with one file at its expected size
  and another at the wrong size, assert the planner marks the correct one as
  "already present, skip" and the other as "needs download," and that `bytesCompleted`
  accounting adds the skipped file's bytes.
- **Total-bytes accounting:** assert `bytesTotal` equals the sum of the filtered
  file sizes from the fixture.

**Prior art:** `parakeet-transcribe/Tests/parakeet-transcribeTests/WindowPlannerTests.swift`
(pure-logic XCTest over a deterministic planner) is the pattern to follow for style
and structure.

## Out of Scope

- Any change to the transcription core: `ModelRunner`, `Transcriber`,
  `StreamingTranscriber`, `TdtDecoder`, `ParakeetTokenizer`, `WindowPlanner`, mel/encoder
  contracts. This spec only changes **where** `modelsDir` points and adds a phase in
  front of load.
- The `parakeet-transcribe` **CLI**: it keeps its `--models <path>` argument and its
  bundled/local-path model loading. No CLI download support.
- Background/off-screen downloading (`URLSession` background configuration), download
  while the app is suspended, or push-to-resume. First-launch download is foreground
  only, gated behind the progress screen.
- Model integrity verification beyond size-match (no SHA/hash checksum of downloaded
  weights); the sentinel + per-file size check is the completeness guarantee.
- Model versioning / update flow: detecting that the remote repo advanced and
  re-downloading. The revision is pinned to `main` and installed once; refreshing means
  clearing the install root (not specified here).
- Delta/patch downloads, mirror/CDN selection, or authenticated (gated) HF repos —
  the repo is public.
- A user-facing setting to choose model quantization or to relocate the download.

## Further Notes

- **Single-seam property.** `TranscriptionEngine.bundledModelsDir()` is the only place
  in the production path that resolves where models live; everything else is
  `modelsDir: URL`-driven. Redirecting that one function to the downloaded root, plus
  the new `.downloading` phase for the UI, is the entire production surface. This is
  the "ideal one seam" the process asks for.
- **Runtime-only repo, defensive filter.** The HF repo was trimmed to exactly the
  runtime set — the raw-logit `parakeet_joint_logits_single_step.mlmodelc` was removed
  (the app uses `joint_decision_single_step`, which argmaxes internally — see
  swift-port-spec §1.1). The allowlist filter is retained anyway so a future stray
  artifact isn't downloaded by accident.
- **Verified repo contents** (HF tree API, `main`): 4 runtime `.mlmodelc` trees +
  `parakeet_vocab.json` (~613 MB after filtering), dominated by the encoder weight
  (`parakeet_encoder.mlmodelc/weights/weight.bin` ≈ 594 MB).
- **Reference implementation** to port from:
  `/Users/sdesai/Tools/MLX/Magpie/Sources/MagpieTTS/ModelDownloader.swift`. It already
  handles the flat-repo layout (install root = both source dir consumers), resume,
  sentinel, and progress — the adaptations are the repo id, the root folder name, and
  the runtime-set allowlist.
- **Deployment target** is iOS 27.0 (`project.yml`); `URLSession.bytes(for:)` and the
  Swift concurrency used by the reference downloader are available.
