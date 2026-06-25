# ANE onset-collapse → empty output (and the CPU-retry fix)

## Symptom

A small number of clips transcribe correctly on CPU but return an **empty
string** on the default `.all` (ANE) compute path. Observed on exactly **1 of
1,372** real benchmark clips:

- `4483857_chunk_213.wav` (earnings22-kws) — 15 s, an earnings call that opens
  *softly and gradually* ("The conference is in presentation mode…") after
  ~6 s of leading near-silence.

No partial collapses (short-but-nonempty outputs) were seen across lsc (9),
FDA-extended (600), or earnings22 (772).

## Root cause: fp16 flips a near-tie at the speech onset, then the greedy
## decode cascades

The TDT predictor (decoder LSTM) only advances on **non-blank** tokens, so
through leading silence it stays frozen at its start-of-sequence (SOS) state.
The **first** emitted token is what "unlocks" it. Until that happens, every
frame is scored against the same stale SOS state.

When speech begins softly, the onset frame's joint decision is a near-tie
between the first real token and blank. Measured on `chunk_213` with the
predictor pinned at SOS, sweeping all 188 encoder frames:

| encoder device | frames that would emit a token (vs SOS) |
|----------------|------------------------------------------|
| CPU (fp32)     | 2 (first at frame 33, token 155)         |
| ANE (fp16)     | **0**                                    |

So even on CPU only 2 of 188 frames clear the blank threshold against SOS — the
onset is marginal across the entire clip. fp16 rounding in the **encoder**
(not the joint) is enough to push the single decisive frame below threshold:

```
t=33 (onset), decoder = SOS:
  CPUenc + CPUjoint → token 155
  CPUenc + ANEjoint → token 155     ← joint device is irrelevant
  ANEenc + ANEjoint → blank (1024)  ← encoder device flips it
  GPUenc + CPUjoint → blank (1024)  ← cpuAndGPU ALSO flips
```

Encoder |CPU−ANE| at the onset frame is tiny — max ≈ 0.0027 on values of norm
≈ 1.7 (≈ 2.6 % relative over the whole tensor). That is all it takes when the
decision is a tie. The flip to blank means the predictor never unlocks, so
every subsequent frame is also judged against SOS and also comes out blank →
**all-blank decode → empty output.**

### Why only CPU fixes it

`cpuAndGPU` was tested as a possible middle ground (it's fp32-ish and keeps the
encoder off the CPU). It **also flips** — its divergence from CPU is half of
ANE's (max 0.0016 vs 0.0027) but the onset is tied tightly enough that any
non-fp32 path tips it. Only `CPU_ONLY` (true fp32) holds the onset above
threshold.

## Does this affect a person speaking into a microphone?

The trigger is **soft/gradual speech onset after silence**, not loudness and
not the recording path. Most human speech begins with an energetic attack (a
plosive or stressed syllable) that emits well above the blank threshold, so
fp16 noise can't flip it — which is why 1,371 of 1,372 real clips are fine. A
speaker who pauses then resumes *very softly/hesitantly* could reproduce it.
Note this is a file/long-form transcriber (record-then-transcribe), not live
streaming. A VAD front-end would trim leading silence but would **not** remove
the risk, because the soft-onset-vs-SOS marginality is intrinsic.

## The fix: empty-output → CPU retry (`Transcriber.transcribe`)

An all-blank decode is an unambiguous signal. When the primary (ANE/GPU) decode
returns **zero tokens** and the primary path isn't already CPU, re-run that
clip's encode+decode once on a lazily-built `CPU_ONLY` runner.

- Triggers only on the rare empty case (~0.07 % of clips), so ANE speed is
  preserved for everything else.
- It is the only proven fix (GPU ruled out; forcing the encoder onto CPU for
  *every* clip would tax the heaviest model to fix one edge case).
- Self-targeting and path-independent: it doesn't need to predict which clips
  are at risk — any future soft-onset collapse is caught by the same
  `tokens.isEmpty` guard.
- No false-positive cost: a genuinely silent clip would also be empty on CPU,
  so the retry just confirms it.

## Reproduce / diagnose

The `PARAKEET_CU_<modelname>` env override (see `ModelRunner.swift`) forces an
individual model onto a chosen device, which is how the encoder was isolated as
the culprit:

```sh
# Force just the encoder onto CPU (everything else stays on .all) — fixes it:
PARAKEET_CU_parakeet_encoder=cpu \
  parakeet-transcribe --models <dir> --audio 4483857_chunk_213.wav

# Forcing the joint or decoder to CPU does NOT help:
PARAKEET_CU_parakeet_joint_decision_single_step=cpu  ...   # still empty
```
