"""Measure transcript accuracy and warm end-to-end latency on a pinned corpus.

Manifest schema::

    {"schema_version": 1, "cases": [{
      "id": "clean-speech", "media": "data/clean.wav",
      "media_sha256": "...", "reference": "data/clean.txt",
      "duration_s": 60.0, "language": "en"
    }]}

Run with ``--backend local`` on a model host or ``--backend remote`` against a
running transcript-server. Accuracy benchmarking intentionally disables word
alignment and diarization so those optional stages do not change the ASR test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    media: Path
    reference: str
    duration_s: float
    language: str


def normalize_text(text: str) -> str:
    """Frozen v1 normalizer used for both references and hypotheses."""
    text = unicodedata.normalize("NFKC", text).casefold()
    normalized = []
    for char in text:
        if char in "'\N{RIGHT SINGLE QUOTATION MARK}":
            continue
        normalized.append(char if unicodedata.category(char)[:1] in {"L", "M", "N"} else " ")
    return " ".join("".join(normalized).split())


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    """Levenshtein distance with linear memory."""
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        pass
    else:
        return Levenshtein.distance(reference, hypothesis)
    # ponytail: O(m*n) is fine beside model inference; use a native scorer if it
    # becomes measurable on very large, single-file references.
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str) -> dict:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    if not reference_words:
        raise ValueError("reference must contain at least one normalized word")
    word_errors = edit_distance(reference_words, hypothesis_words)
    reference_chars = normalized_reference.replace(" ", "")
    hypothesis_chars = normalized_hypothesis.replace(" ", "")
    char_errors = edit_distance(reference_chars, hypothesis_chars)
    return {
        "wer": word_errors / len(reference_words),
        "cer": char_errors / len(reference_chars),
        "word_errors": word_errors,
        "reference_words": len(reference_words),
        "char_errors": char_errors,
        "reference_chars": len(reference_chars),
        "normalized_hypothesis_sha256": hashlib.sha256(
            normalized_hypothesis.encode("utf-8")
        ).hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(manifest_path: Path) -> list[Case]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest cases must be a non-empty list")

    base = manifest_path.resolve().parent
    cases = []
    seen = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("every manifest case must be an object")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"case id must be non-empty and unique: {case_id!r}")
        seen.add(case_id)
        media = (base / str(raw.get("media", ""))).resolve()
        reference_path = (base / str(raw.get("reference", ""))).resolve()
        if not media.is_file() or not reference_path.is_file():
            raise ValueError(f"case {case_id!r} media/reference file is missing")
        expected_digest = raw.get("media_sha256")
        if not isinstance(expected_digest, str) or _sha256(media) != expected_digest.lower():
            raise ValueError(f"case {case_id!r} media_sha256 does not match")
        duration_s = raw.get("duration_s")
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or duration_s <= 0):
            raise ValueError(f"case {case_id!r} duration_s must be positive and finite")
        language = raw.get("language")
        if not isinstance(language, str) or not language:
            raise ValueError(f"case {case_id!r} language must be non-empty")
        reference = reference_path.read_text(encoding="utf-8")
        if not normalize_text(reference):
            raise ValueError(f"case {case_id!r} reference has no normalized words")
        cases.append(Case(case_id, media, reference, float(duration_s), language))
    return cases


def summarize(rows: list[dict], cases: list[Case], runs: int) -> dict:
    total_audio_s = sum(case.duration_s for case in cases)
    run_totals = [sum(row["elapsed_s"] for row in rows if row["run"] == run)
                  for run in range(1, runs + 1)]
    elapsed_s = statistics.median(run_totals)
    return {
        "cases": len(cases),
        "runs": runs,
        "corpus_wer": sum(row["word_errors"] for row in rows)
        / sum(row["reference_words"] for row in rows),
        "corpus_cer": sum(row["char_errors"] for row in rows)
        / sum(row["reference_chars"] for row in rows),
        "median_corpus_elapsed_s": elapsed_s,
        "realtime_factor": elapsed_s / total_audio_s,
        "x_realtime": total_audio_s / elapsed_s,
    }


def gate_failures(summary: dict, *, max_wer: float | None,
                  max_rtf: float | None) -> list[str]:
    failures = []
    if max_wer is not None and summary["corpus_wer"] > max_wer:
        failures.append(f"WER {summary['corpus_wer']:.2%} exceeds {max_wer:.2%}")
    if max_rtf is not None and summary["realtime_factor"] > max_rtf:
        failures.append(f"RTF {summary['realtime_factor']:.3f} exceeds {max_rtf:.3f}")
    return failures


def _run_local(case: Case, engine) -> tuple[str, dict, dict]:
    from transcript import transcribe

    with tempfile.TemporaryDirectory(prefix="transcript-accuracy-") as work_dir:
        started = time.perf_counter()
        result = transcribe(
            str(case.media), engine=engine, language=case.language,
            diarize=False, align=False, work_dir=work_dir,
        )
        elapsed = time.perf_counter() - started
    return result.text, {"elapsed_s": elapsed}, result.meta


def _run_remote(case: Case, *, server: str, token: str | None,
                poll: float, timeout: float) -> tuple[str, dict, dict]:
    import requests

    from transcript._remote_http import (
        build_headers, get_with_retry, poll_until_done, request_timeout, submit_job,
    )

    base = server.rstrip("/")
    headers = build_headers(token)
    started = time.perf_counter()
    upload_started = time.perf_counter()
    job_id = None
    try:
        with case.media.open("rb") as media:
            job_id = submit_job(
                requests, f"{base}/jobs",
                data={"diarize": "false", "align": "false", "language": case.language},
                files={"file": (case.media.name, media)}, headers=headers, timeout=timeout,
            )
        upload_s = time.perf_counter() - upload_started
        status = poll_until_done(
            requests, f"{base}/jobs/{job_id}", headers,
            poll=poll, timeout=timeout, note=lambda _message: None,
        )
        response = get_with_retry(
            requests, f"{base}/jobs/{job_id}/result", params={"format": "json"},
            headers=headers, deadline=time.monotonic() + timeout,
            operation="fetching benchmark result",
        )
        if not response.ok:
            raise RuntimeError(
                f"fetching result failed ({response.status_code}): {response.text[:200]}"
            )
        payload = response.json()
        hypothesis = "\n".join(
            segment.get("text", "").strip() for segment in payload.get("segments", [])
            if segment.get("text", "").strip()
        )
        server_s = None
        if status.get("started_at") is not None and status.get("finished_at") is not None:
            server_s = status["finished_at"] - status["started_at"]
        return hypothesis, {
            "elapsed_s": time.perf_counter() - started,
            "upload_s": upload_s,
            "server_s": server_s,
        }, payload.get("meta", {})
    finally:
        if job_id is not None:
            try:
                cleanup = requests.delete(
                    f"{base}/jobs/{job_id}", headers=headers,
                    timeout=request_timeout(timeout),
                )
                if not cleanup.ok and cleanup.status_code != 404:
                    print(
                        f"warning: could not delete benchmark job {job_id} "
                        f"({cleanup.status_code})",
                        file=sys.stderr,
                    )
            except requests.RequestException as exc:
                print(f"warning: could not delete benchmark job {job_id}: {exc}", file=sys.stderr)


def run_benchmark(args, cases: list[Case]) -> dict:
    engine = None
    if args.backend == "local":
        from transcript import TranscriptionEngine

        engine = TranscriptionEngine(
            model=args.model, device=args.device, compute_type=args.compute_type,
            batch_size=args.batch_size,
        )

    def run_case(case: Case):
        if args.backend == "local":
            return _run_local(case, engine)
        return _run_remote(
            case, server=args.server, token=os.environ.get("TRANSCRIPT_TOKEN"),
            poll=args.poll, timeout=args.timeout,
        )

    for _ in range(args.warmups):
        run_case(cases[0])

    rows = []
    metadata = {}
    for run in range(1, args.runs + 1):
        for case in cases:
            hypothesis, timings, metadata = run_case(case)
            row = {
                "case": case.id,
                "run": run,
                "duration_s": case.duration_s,
                **timings,
                **score(case.reference, hypothesis),
                "hypothesis": hypothesis,
            }
            row["realtime_factor"] = row["elapsed_s"] / case.duration_s
            rows.append(row)
            print(
                f"{case.id} run={run}: WER={row['wer']:.2%} "
                f"time={row['elapsed_s']:.3f}s RTF={row['realtime_factor']:.3f}",
                flush=True,
            )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "configuration": {
            "model": metadata.get("model", args.model),
            "device": metadata.get("device", args.device),
            "compute_type": metadata.get("compute_type", args.compute_type),
            "batch_size": args.batch_size if args.backend == "local" else None,
            "align": False,
            "diarize": False,
        },
        "system": {
            "role": "local_host" if args.backend == "local" else "client_host",
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "summary": summarize(rows, cases, args.runs),
        "results": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=("local", "remote"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3", help="Local backend model.")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-wer", type=float, help="Fail when corpus WER exceeds this ratio.")
    parser.add_argument("--max-rtf", type=float, help="Fail when real-time factor exceeds this ratio.")
    parser.add_argument(
        "--server", default=os.environ.get("TRANSCRIPT_SERVER", "http://localhost:8000")
    )
    parser.add_argument("--poll", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1 or args.warmups < 0 or args.batch_size < 1:
        raise SystemExit("runs and batch-size must be positive; warmups cannot be negative")
    if any(value is not None and (not math.isfinite(value) or value < 0)
           for value in (args.max_wer, args.max_rtf)):
        raise SystemExit("max-wer and max-rtf must be finite, non-negative ratios")
    cases = load_cases(args.manifest)
    report = run_benchmark(args, cases)
    report["manifest_sha256"] = _sha256(args.manifest)
    failures = gate_failures(
        report["summary"], max_wer=args.max_wer, max_rtf=args.max_rtf,
    )
    report["gates"] = {
        "max_wer": args.max_wer,
        "max_rtf": args.max_rtf,
        "passed": not failures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
