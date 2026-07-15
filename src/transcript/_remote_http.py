"""Shared HTTP/poll helpers for the thin clients (``transcript-remote`` and
``extract-remote``). Only ``requests`` + stdlib — no torch/whisperx/zip-heavy deps.

Factored out so both clients share one submit/poll implementation instead of
drifting apart.
"""

from __future__ import annotations

import math
import re
import sys
import time
from typing import Callable, Optional


_JOB_ID_RE = re.compile(r"[0-9a-f]{12}\Z")


def build_headers(token: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def stderr_note(quiet: bool) -> Callable[[str], None]:
    def note(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr)
    return note


def request_timeout(seconds: float) -> tuple[float, float]:
    """Requests connect/read timeout derived from the user's overall limit."""
    return min(30.0, seconds), seconds


def response_job_id(response) -> str:
    """Read and validate the untrusted job id returned by a submit route."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("server returned malformed JSON when submitting the job") from exc
    job_id = str(payload.get("id", "")) if isinstance(payload, dict) else ""
    if not _JOB_ID_RE.fullmatch(job_id):
        raise RuntimeError(f"server returned an invalid job id: {job_id!r}")
    return job_id


def validate_common_options(
    *,
    poll: float,
    timeout: float,
    diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Optional[str]:
    if not math.isfinite(timeout) or timeout <= 0:
        return "--timeout must be greater than zero"
    if not math.isfinite(poll) or poll <= 0:
        return "--poll must be greater than zero"
    if min_speakers is not None and min_speakers < 1:
        return "--min-speakers must be at least 1"
    if max_speakers is not None and max_speakers < 1:
        return "--max-speakers must be at least 1"
    if (min_speakers is not None and max_speakers is not None
            and min_speakers > max_speakers):
        return "--min-speakers cannot exceed --max-speakers"
    if not diarize and (min_speakers is not None or max_speakers is not None):
        return "speaker-count hints cannot be used with --no-diarize"
    return None


def poll_until_done(
    requests_mod,
    status_url: str,
    headers: dict,
    *,
    poll: float,
    timeout: float,
    note: Callable[[str], None],
) -> dict:
    """Poll ``status_url`` until the job is ``done`` (or raise on error/timeout).

    Returns the final status dict. Raises :class:`RuntimeError` with a clean
    message on a failed job or a timeout.
    """
    # Measure against the monotonic wall clock so slow network requests count
    # toward the deadline (accumulating only `poll` would overshoot `timeout`).
    deadline = time.monotonic() + timeout
    last_status = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"timed out after {timeout}s.")
        try:
            resp = requests_mod.get(
                status_url, headers=headers, timeout=request_timeout(remaining),
            )
            # A non-2xx status (401/404/500…) is a real server error — surface it
            # instead of treating its JSON body as a job with status=None and
            # looping until the overall timeout.
            if not resp.ok:
                raise RuntimeError(f"server returned {resp.status_code}: {resp.text[:200]}")
            s = resp.json()
        except (requests_mod.RequestException, ValueError) as exc:
            # ValueError covers a non-JSON body (e.g. a proxy 502 HTML page) whose
            # .json() raises JSONDecodeError — surface it as a clean RuntimeError.
            raise RuntimeError(f"Error polling job: {exc}") from exc
        if not isinstance(s, dict):
            raise RuntimeError("Error polling job: server returned malformed JSON")

        status = s.get("status")
        if status != last_status:
            note(f"  status: {status}")
            last_status = status

        if status == "done":
            return s
        if status == "error":
            reason = f" [{s['error_reason']}]" if s.get("error_reason") else ""
            raise RuntimeError(f"job failed{reason}: {s.get('error') or 'unknown error'}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"timed out after {timeout}s (job still {status}).")
        time.sleep(min(poll, remaining))
