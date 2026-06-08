"""Shared HTTP/poll helpers for the thin clients (``transcript-remote`` and
``extract-remote``). Only ``requests`` + stdlib — no torch/whisperx/zip-heavy deps.

Factored out so both clients share one submit/poll implementation instead of
drifting apart.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional


def build_headers(token: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def stderr_note(quiet: bool) -> Callable[[str], None]:
    def note(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr)
    return note


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
        try:
            resp = requests_mod.get(status_url, headers=headers, timeout=(30, 60))
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

        status = s.get("status")
        if status != last_status:
            note(f"  status: {status}")
            last_status = status

        if status == "done":
            return s
        if status == "error":
            raise RuntimeError(f"job failed: {s.get('error')}")

        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out after {timeout}s (job still {status}).")
        time.sleep(poll)
