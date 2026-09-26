# Creator-reference accuracy calibration

This corpus is separate from the unchanged 40 no-creator-caption regression
videos. The initial four-video pilot checks the reference/scoring pipeline;
the target accuracy corpus is 20 English and 20 Japanese videos.

Freeze creator captions and their corresponding audio on Windows:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy freeze benchmarks\youtube_accuracy\pilot.json C:\Users\leewe\transcript-validation\youtube-accuracy\pilot
.\.venv\Scripts\python.exe -m benchmarks.youtube_baseline.run C:\Users\leewe\transcript-validation\youtube-accuracy\pilot\pinned.json --output C:\Users\leewe\transcript-validation\youtube-accuracy\baseline
.\.venv\Scripts\python.exe -m benchmarks.youtube_accuracy score C:\Users\leewe\transcript-validation\youtube-accuracy\pilot\pinned.json C:\Users\leewe\transcript-validation\youtube-accuracy\baseline
```

Only frozen audio bytes are uploaded to generation. References are never passed
as prompts, supplied for alignment, or retrieved through the URL transcription
path. Raw creator captions, reference text and audio are independently hashed.
Changed references require a recorded new corpus version, never a silent edit.

English uses WER; Japanese uses CER. The existing frozen text normalizer handles
case, punctuation and Unicode width. Initial scores retain creator annotations,
so non-speech labels and edited captions must be examined as possible reference
differences. Creator-provided does not itself establish human authorship,
verbatim completeness, correct speakers, word timing, or spoken-only content.
Those requirements are reported separately, not inferred from WER/CER.

Keep calibration and holdout sets separate. Diagnose calibration mismatches,
test general fixes, then measure the holdout without tuning to its individual
answers. Preserve the original results and use new output directories for
candidate runs. Never import benchmark transcripts into the user's app library.
