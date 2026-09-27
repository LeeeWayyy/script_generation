# Creator-reference accuracy calibration

This corpus is separate from the unchanged 40 no-creator-caption regression
videos. The initial four-video pilot checks the reference/scoring pipeline;
`manifest.json` selects 20 English and 20 Japanese videos, with 14 calibration
and 6 holdout cases per language. Topics include science, history, food,
technology, travel, business and everyday conversation; durations span about
2–57 minutes. Selection used caption/audio metadata, not transcription scores.
The Windows full run is in `youtube-accuracy/full-v1` (frozen inputs) and
`youtube-accuracy/full-baseline` (audio-only API outputs).
`frozen-inputs.json` records the exact audio/reference hashes for all 40 inputs.

Freeze creator captions and their corresponding audio on Windows:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy freeze benchmarks\youtube_accuracy\pilot.json C:\Users\leewe\transcript-validation\youtube-accuracy\pilot
.\.venv\Scripts\python.exe -m benchmarks.youtube_baseline.run C:\Users\leewe\transcript-validation\youtube-accuracy\pilot\pinned.json --output C:\Users\leewe\transcript-validation\youtube-accuracy\baseline
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy score C:\Users\leewe\transcript-validation\youtube-accuracy\pilot\pinned.json C:\Users\leewe\transcript-validation\youtube-accuracy\baseline
```

For a fresh full-corpus reproduction, with the server running and its token in
the environment, use a new output directory:

```powershell
$benchmarkRoot = 'C:\Users\leewe\transcript-validation\youtube-accuracy'
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy freeze benchmarks\youtube_accuracy\manifest.json "$benchmarkRoot\full-v1"
.\.venv\Scripts\python.exe -m benchmarks.youtube_baseline.repeat "$benchmarkRoot\full-v1\pinned.json" "$benchmarkRoot\new-repeat" --runs 2
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy score "$benchmarkRoot\full-v1\pinned.json" "$benchmarkRoot\new-repeat\run-1"
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy score "$benchmarkRoot\full-v1\pinned.json" "$benchmarkRoot\new-repeat\run-2"
```

Reuse the frozen audio for exact reproduction: a new YouTube download may differ.
Compare its hashes to `frozen-inputs.json` before comparing runs. The repeat test
performs new inference; it ignores only execution-specific source/job identifiers.
It does not restart the server or establish cross-machine/cross-version identity.

Only frozen audio bytes are uploaded to generation. References are never passed
as prompts, supplied for alignment, or retrieved through the URL transcription
path. Raw creator captions, reference text and audio are independently hashed.
Changed references require a recorded new corpus version, never a silent edit.

English uses WER with the installed Whisper English normalizer, retaining currency
and percent units and equating spoken/written number forms. Curly apostrophes are made ASCII so
contractions expand (`it’s` → `it is`), and `ok` is spelled `okay`; the version
and spelling map are recorded. Japanese uses CER with the existing Unicode normalizer.
Version 3 scores remove caption annotations before scoring both texts: bracketed
sounds and implied words (`[LAUGHTER]`, `（私は）`) and line-leading speaker labels
(`PROFESSOR:`). Fillers and backchannels are not required transcript content:
English drops them in the Whisper normalizer, and Japanese scoring removes kana
fillers (`えっと`, `えー`, `あー`, `うん`, `うーん`, `あのー`) from both texts. Each row keeps the strict v1 and annotations-retained v2 scores;
reports are written to `reference-scores-v3.json`, preserving earlier reports.
Groups also report a full-coverage aggregate that omits partial-coverage references,
which otherwise count every later generated word as an insertion. Space-delimited
labels (`ja-reference-20`: `ursさん ...`) are not detectable and remain scored. Non-speech labels,
orthographic choices and edited captions remain possible reference differences.
Creator-provided does not itself establish human authorship,
verbatim completeness, correct speakers, word timing, or spoken-only content.
Those requirements are reported separately, not inferred from WER/CER.
Caption tracks that start after 20% or end before 80% of the media receive a
coverage warning. This is a review screen, not proof that a gap contains speech.
The disaster simulation case `ja-reference-02` has only 203 reference characters
and captions ending near 402 seconds of 807; retain it as a diagnostic, not a
valid full-video accuracy reference. Aggregate scores retain all selected cases,
including flagged ones, and must not be presented as ground-truth ASR accuracy.

Keep calibration and holdout sets separate. Diagnose calibration mismatches,
test general fixes, then measure the holdout without tuning to its individual
answers. Preserve the original results and use new output directories for
candidate runs. Never import benchmark transcripts into the user's app library.

## Initial calibration decisions

`pilot-results.json` retains the four baseline scores and native decoder trials.
The native decoder with previous-text context disabled slightly improved Japanese
calibration CER (2.60% to 2.29%), but worsened Japanese holdout CER (5.39% to 6.33%)
and English calibration WER (7.95% to 8.23%). It was rejected. Beam 10 produced
the same pilot calibration text as beam 5. Previous-text conditioning produced
severe English repetition in the WhisperX subclass experiment and was rejected.
The production decoder remains unchanged.

The KIND business video's creator captions contain suspect wording, recorded in
the manifest and reports. It remains visible as a diagnostic case; its differences
must not be treated as proven generation errors or silently removed from scores.
The full corpus still requires reference-quality review. Neither pilot reference
agreement nor previous reproducibility checks establish acoustic timing, speaker
accuracy, or the spoken-only requirement. Accuracy acceptance remains incomplete.
