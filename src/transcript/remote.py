"""Mac-side client — talk to a transcript-server running on the GPU host.

Needs only `requests` (no torch/whisperx). Submits a local file or URL, polls
until the job finishes, then prints or saves the result.

Examples:
    export TRANSCRIPT_SERVER=http://gpuhost:8000
    export TRANSCRIPT_TOKEN=...                 # if the server requires auth

    transcript-remote meeting.mp4 -f srt -o meeting.srt
    transcript-remote "https://youtube.com/watch?v=..." -f txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._remote_http import build_headers, poll_until_done, stderr_note
from .ingest import is_url


def main(argv: list[str] | None = None) -> int:
    import requests

    p = argparse.ArgumentParser(
        prog="transcript-remote",
        description="Submit a file/URL to a remote transcript-server and fetch the transcript.",
    )
    p.add_argument("source", help="Local media file path or http(s) URL.")
    p.add_argument("-f", "--format", default="txt", choices=["txt", "srt", "vtt", "json"])
    p.add_argument("-o", "--output", help="Write result to this file instead of stdout.")
    p.add_argument(
        "--server",
        default=os.environ.get("TRANSCRIPT_SERVER", "http://localhost:8000"),
        help="Server base URL (or $TRANSCRIPT_SERVER).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("TRANSCRIPT_TOKEN"),
        help="Bearer token if the server requires auth (or $TRANSCRIPT_TOKEN).",
    )
    p.add_argument("--no-diarize", dest="diarize", action="store_false", default=True)
    p.add_argument("--language", help="Force language code (e.g. en).")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--poll", type=float, default=3.0, help="Seconds between status checks.")
    p.add_argument("--timeout", type=float, default=3600.0, help="Give up after N seconds.")
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Show progress on stderr (the default)."
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress on stderr.")
    args = p.parse_args(argv)

    base = args.server.rstrip("/")
    headers = build_headers(args.token)
    note = stderr_note(args.quiet)

    # Build the multipart form (works for both upload and URL).
    data = {"diarize": str(args.diarize).lower()}
    if args.language:
        data["language"] = args.language
    if args.min_speakers is not None:
        data["min_speakers"] = str(args.min_speakers)
    if args.max_speakers is not None:
        data["max_speakers"] = str(args.max_speakers)

    files = None
    fh = None
    try:
        if is_url(args.source):
            data["url"] = args.source
            note(f"Submitting URL to {base} ...")
        else:
            path = Path(args.source).expanduser()
            if not path.is_file():
                print(f"Error: file not found: {path}", file=sys.stderr)
                return 1
            fh = path.open("rb")
            files = {"file": (path.name, fh)}
            note(f"Uploading {path.name} to {base} ...")

        try:
            r = requests.post(f"{base}/jobs", data=data, files=files, headers=headers)
        finally:
            if fh:
                fh.close()
    except requests.RequestException as exc:
        print(f"Error: could not reach server at {base}: {exc}", file=sys.stderr)
        return 1

    if r.status_code == 401:
        print("Error: unauthorized. Set --token / $TRANSCRIPT_TOKEN.", file=sys.stderr)
        return 1
    if not r.ok:
        print(f"Error: server rejected job ({r.status_code}): {r.text}", file=sys.stderr)
        return 1

    job_id = r.json()["id"]
    note(f"Job {job_id} queued. Polling ...")

    # Shared submit/poll logic (also used by extract-remote) lives in _remote_http.
    try:
        poll_until_done(requests, f"{base}/jobs/{job_id}", headers,
                        poll=args.poll, timeout=args.timeout, note=note)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rr = requests.get(
        f"{base}/jobs/{job_id}/result", params={"format": args.format}, headers=headers,
        timeout=(30, args.timeout),  # don't hang forever on a stalled server
    )
    if not rr.ok:
        print(f"Error fetching result ({rr.status_code}): {rr.text}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).expanduser().write_text(rr.text, encoding="utf-8")
        note(f"Wrote {args.format} -> {args.output}")
    else:
        sys.stdout.write(rr.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
