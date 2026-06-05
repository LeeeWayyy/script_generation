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
    waited = 0.0
    last_status = None
    while True:
        try:
            s = requests_mod.get(status_url, headers=headers).json()
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

        time.sleep(poll)
        waited += poll
        if waited >= timeout:
            raise RuntimeError(f"timed out after {timeout}s (job still {status}).")
