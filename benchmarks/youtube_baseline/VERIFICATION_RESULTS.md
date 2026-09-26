# Windows transcript verification

**Full accuracy acceptance: NOT PASSED.** Spoken words only; sung lyrics are excluded by the requirement, but complete exclusion has not been independently verified.

Three runs of the same 40 frozen inputs (20 English, 20 Japanese; 10.00 hours per run), each beginning in a fresh Windows server process. Inference code: `4d0b537a40dc997261b881fac8811372df8688d4`.

- 37/40 cases have exactly matching raw and readable transcript content across all three runs.
- 3/40 cases consistently return the no-reliable-speech error without publishing captions. These are not successful transcripts or independently certified non-speech inputs.
- All 37 generated transcripts in pass one pass the unchanged production app parser; all 134,073 supplied aligned entries produce usable cues. The same app checks were run separately on passes two and three; see the per-case evidence.
- Comparison includes text, word times, speaker labels, scores, and metadata in both file formats. Only job IDs and execution source paths are excluded.
- Every input is verified by SHA-256 before upload. No creator-caption tracks were present at freezing; all non-null selected audio-language labels match the requested language. Missing labels and topic/type eligibility still require content review.

## Accuracy remains unresolved

The 7,604 generated reading rows have no independent listening-reviewed references. There are 396 aligned entries with no speaker label and 252 rows with unknown or mixed speaker attribution. Valid timing fields do not establish acoustic timing accuracy.

Suspected Japanese recognition errors remain in the short science example, including homophones around 85–92 and 113–122 seconds. The sports example at 33–42 seconds still needs independent speaker review. No phrase was rewritten to match expectations, and no speaker was assigned merely to eliminate an unknown label.

The speech detector suppresses the three previously problematic music-heavy outputs in this corpus. It is not a general singing classifier and does not prove that mixed speech/music outputs contain no lyrics. Correct words, spoken-versus-sung classification, true speaker turns, acoustic timing, and complete sentence/expression boundaries remain open for every unreviewed row. No WER/CER or zero-error claim is supported.

## Changes verified

- Use the versioned Silero weights already bundled with faster-whisper for speech detection; refuse empty or unsupported speech results.
- Remove an alignment boundary entry only when it is demonstrably duplicated outside the left source text and belongs to the adjacent right segment. The retained provenance records this adjustment.
- Disable CUDA matrix and cuDNN TF32 before the first alignment. Previously, diarization changed precision only after that first alignment, producing different confidence scores on a fresh process. The sensitive first-job case is included in every cold pass.
- Use a fresh connection for each benchmark upload. No upload is automatically retried, so uncertain requests cannot silently create duplicate jobs.
- Resolve the actual Deno executable on Windows instead of the broken WinGet link. A fresh request for the original YouTube interview completed successfully; its “I made it up.” row remains intact at 93.767–94.287 seconds.

## Reproduction and evidence

The SSH-attached benchmark driver exited after 16 saved cases in pass three. It resumed from those files in a persistent logged process; the inference server PID remained unchanged. No completed transcript was regenerated or replaced during recovery.

`verification_summary.json` records per-case input hashes, all job IDs, raw/readable content hashes, app results, model-file hashes, package versions, precision flags, and cold-server records. Existing jobs and raw results were snapshotted and verified unchanged after each restart. Reproduction is demonstrated on this host and these pinned model files; it is not a claim of identical output across arbitrary hardware or dependency versions.

Windows audio: `C:\Users\leewe\transcript-validation\youtube-baseline\frozen`.

Windows three-pass results: `C:\Users\leewe\transcript-validation\youtube-baseline\precision-repeatability`.

Use the freeze/repeat commands in the baseline README with a **new output directory** for fresh inference. Retain the pinned audio, the recorded model files, and runtime versions. Reusing a completed output directory resumes saved evidence rather than creating new jobs.

| Case | Raw/readable exact match | Result |
|---|---|---|
| en-science-short | — | repeated error |
| en-science-medium | yes | transcript generated; accuracy unreviewed |
| en-history-short | yes | transcript generated; accuracy unreviewed |
| en-history-long | yes | transcript generated; accuracy unreviewed |
| en-cooking-short | yes | transcript generated; accuracy unreviewed |
| en-cooking-medium | yes | transcript generated; accuracy unreviewed |
| en-technology-medium | yes | transcript generated; accuracy unreviewed |
| en-technology-long | yes | transcript generated; accuracy unreviewed |
| en-travel-medium | — | repeated error |
| en-sports-short | yes | transcript generated; accuracy unreviewed |
| en-sports-medium | yes | transcript generated; accuracy unreviewed |
| en-arts-medium | yes | transcript generated; accuracy unreviewed |
| en-arts-long | yes | transcript generated; accuracy unreviewed |
| en-business-medium | yes | transcript generated; accuracy unreviewed |
| en-business-long | yes | transcript generated; accuracy unreviewed |
| en-education-extended | yes | transcript generated; accuracy unreviewed |
| en-culture-medium | yes | transcript generated; accuracy unreviewed |
| en-culture-extended | yes | transcript generated; accuracy unreviewed |
| ja-science-short | yes | transcript generated; accuracy unreviewed |
| ja-science-medium | yes | transcript generated; accuracy unreviewed |
| ja-history-short | yes | transcript generated; accuracy unreviewed |
| ja-history-long | yes | transcript generated; accuracy unreviewed |
| ja-cooking-short | yes | transcript generated; accuracy unreviewed |
| ja-cooking-medium | yes | transcript generated; accuracy unreviewed |
| ja-technology-medium | yes | transcript generated; accuracy unreviewed |
| ja-technology-long | yes | transcript generated; accuracy unreviewed |
| ja-travel-medium | — | repeated error |
| ja-sports-short | yes | transcript generated; accuracy unreviewed |
| ja-sports-medium | yes | transcript generated; accuracy unreviewed |
| ja-arts-long | yes | transcript generated; accuracy unreviewed |
| ja-business-medium | yes | transcript generated; accuracy unreviewed |
| ja-business-long | yes | transcript generated; accuracy unreviewed |
| ja-education-medium | yes | transcript generated; accuracy unreviewed |
| ja-education-extended | yes | transcript generated; accuracy unreviewed |
| ja-culture-medium | yes | transcript generated; accuracy unreviewed |
| ja-culture-extended | yes | transcript generated; accuracy unreviewed |
| en-travel-short | yes | transcript generated; accuracy unreviewed |
| en-education-medium | yes | transcript generated; accuracy unreviewed |
| ja-travel-short | yes | transcript generated; accuracy unreviewed |
| ja-arts-medium | yes | transcript generated; accuracy unreviewed |
