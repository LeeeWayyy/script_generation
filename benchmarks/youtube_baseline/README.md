# English/Japanese YouTube acceptance baseline

Run on the Windows GPU host against the existing authenticated service:

```powershell
.\.venv\Scripts\python.exe benchmarks\youtube_baseline\run.py benchmarks\youtube_baseline\manifest.json --output C:\Users\leewe\transcript-validation\youtube-baseline\results
```

`TRANSCRIPT_TOKEN` is read from the environment and never saved. One full video is
submitted at a time with alignment and diarization enabled. Completed jobs and raw
and readable JSON are retained. The runner resumes from saved job IDs/results;
it never restarts the service or deletes an existing job.

Selection: 20 English and 20 Japanese public videos, ten topic groups per
language, short (1–5 min), medium (5–15), long (15–30), and extended (30–60).
Creator-provided captions in **any** language are excluded at discovery time.
Automatic captions may exist but are never used for generation or scoring.
Availability and caption status can change: recheck before future comparisons.
Topic and format labels are selection labels and require content review.

The immutable first run is a baseline, not a claim that every case passes.
Keep failed results and candidate mistakes visible. Do not drop a failing case,
loosen a gate, or silently replace it to improve the success rate. Corrections
must be rerun against the same video and compared with the retained baseline.

## Acceptance means all requirements, for every row

- Successful full audio-based inference, alignment and diarization; no caption reuse.
- Spoken words only: sung lyrics and invented speech during music are failures.
- Unchanged speech/word metadata between raw and readable views.
- Production app JSON limits, nondecreasing starts, valid bounds, unique in-range
  fallback indices, and valid fallback source bounds.
- Production app parser and word-cue checks, using unchanged app code separately.
- Correct words, speaker turns, sentence/expression continuity, and acoustic timing
  require independent listening-based references for **every** affected row.
  Model success flags, safe fallbacks, short-row heuristics, another ASR model,
  or agreement with automatic captions do not certify those requirements.

`functional_pass` reports only the executable structural/inference gates.
`acceptance_pass` remains false until independent review requirements are resolved.
Missing timing, unknown speakers, short rows, repeated text and truncated coverage
are review flags, never waived by a corpus average. No WER/CER number is emitted
without reviewed reference text. Reuse `benchmarks/run.py` for pinned-reference
WER/CER once those references exist; Japanese accuracy uses CER primarily.

History, arts and education form a six-video holdout per language. Other topics
form the calibration subset. Do not tune rules to individual video IDs or fix
transcript text by hard-coded substitutions. Freeze input metadata, code/version
provenance and all first-run artifacts before comparing general algorithm changes.

## Reassess and check the actual app

```sh
python -m benchmarks.youtube_baseline.report benchmarks/youtube_baseline/manifest.json benchmarks/youtube_baseline/results
swiftc -parse-as-library -swift-version 5 "$SCRIPT_VIEWING_ROOT/App/Models.swift" "$SCRIPT_VIEWING_ROOT/App/TranscriptPresentation.swift" benchmarks/youtube_baseline/check_app.swift -o /tmp/check-baseline-app
/tmp/check-baseline-app benchmarks/youtube_baseline/results/*.readable.json > benchmarks/youtube_baseline/results/app-parser-checks.json
```

The Swift adapter calls production parsing and cue logic. No client validation is
weakened. It catches individual case failures so all cases are reported, and
counts aligned entries versus usable cues instead of treating one cue per row
as complete timing coverage. Preserve the app source revision alongside results.

Compare a candidate reading algorithm against the frozen first run without
changing text, timestamps, word labels, or baseline files:

```sh
PYTHONPATH=src python -m benchmarks.youtube_baseline.compare benchmarks/youtube_baseline/manifest.json benchmarks/youtube_baseline/results benchmarks/youtube_baseline/results/candidate
```

Use a different output directory for a fresh inference run. Existing completed
cases are reused, including failures; a rerun must not erase original evidence.

## Reproduce inference from identical audio

Run these modules from the repository root on the GPU host:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.youtube_baseline.freeze benchmarks\youtube_baseline\manifest.json C:\Users\leewe\transcript-validation\youtube-baseline\frozen
.\.venv\Scripts\python.exe -m benchmarks.youtube_baseline.repeat C:\Users\leewe\transcript-validation\youtube-baseline\frozen\pinned.json C:\Users\leewe\transcript-validation\youtube-baseline\repeatability
```

The first command retains decoded 16-kHz mono PCM audio and SHA-256 hashes,
and rechecks creator-caption eligibility. Resuming verifies existing audio;
changed inputs are rejected. Keep this corpus directory with the results.
Ensure the actual Deno executable is on PATH; a broken Windows WinGet link
does not provide a working JavaScript runtime.

The second command uploads the same verified bytes for three new inference
passes. It compares all caption text, word data, timing, speaker labels and
metadata, excluding execution IDs and source paths. An identical error is
reported separately and never counts as successful caption generation.
Repeatability does not establish accuracy; both gates must pass independently.
