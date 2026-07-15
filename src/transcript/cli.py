"""Command-line interface: ``transcript <source> [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, transcribe
from .engine import DEFAULT_MODEL
from .formats import FORMATS, render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript",
        description="Generate a transcript (optionally with speaker labels) from any "
        "video/audio file or URL.",
    )
    p.add_argument("source", help="Local media file path or http(s) URL (YouTube, etc.).")
    p.add_argument(
        "-f",
        "--format",
        default="txt",
        choices=FORMATS,
        help="Output format (default: txt).",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Write to this file instead of stdout. Extension is not auto-changed.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL}). e.g. large-v3, medium, small.",
    )
    p.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        default=True,
        help="Disable speaker identification (default: on). Needs a Hugging Face token.",
    )
    p.add_argument("--language", help="Force language code (e.g. en). Default: auto-detect.")
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Force device. Default: auto (CUDA if available, else CPU).",
    )
    p.add_argument("--compute-type", help="CTranslate2 compute type (e.g. float16, int8).")
    p.add_argument("--hf-token", help="Hugging Face token (else uses $HF_TOKEN).")
    p.add_argument("--min-speakers", type=int, help="Minimum number of speakers (diarization hint).")
    p.add_argument("--max-speakers", type=int, help="Maximum number of speakers (diarization hint).")
    p.add_argument("--batch-size", type=int, default=16, help="ASR batch size (default: 16).")
    p.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        default=True,
        help="Skip word-level alignment (faster, coarser timestamps).",
    )
    p.add_argument(
        "--detect-music",
        action="store_true",
        help="Flag transcript segments overlapping music (default: off).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = transcribe(
            args.source,
            model=args.model,
            diarize=args.diarize,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            batch_size=args.batch_size,
            align=args.align,
            detect_music=args.detect_music,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # surface a clean message, full trace only when -v
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = render(result, args.format)

    if args.output:
        Path(args.output).expanduser().write_text(output, encoding="utf-8")
        speakers = f" | speakers: {', '.join(result.speakers)}" if result.speakers else ""
        print(
            f"Wrote {args.format} -> {args.output} "
            f"({len(result.segments)} segments{speakers})",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
