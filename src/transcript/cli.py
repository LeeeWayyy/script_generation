"""Command-line interface: ``transcript <source> [options]``."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from . import TranscriptionEngine, __version__, transcribe
from .engine import DEFAULT_MODEL
from .formats import FORMATS, render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcript",
        description="Generate a transcript (optionally with speaker labels) from any "
        "video/audio file or URL.",
    )
    p.add_argument(
        "source", nargs="+", help="One or more local media paths or http(s) URLs."
    )
    p.add_argument(
        "-f",
        "--format",
        default="txt",
        choices=FORMATS,
        help="Output format (default: txt).",
    )
    destination = p.add_mutually_exclusive_group()
    destination.add_argument(
        "-o",
        "--output",
        help="Write to this file instead of stdout. Extension is not auto-changed.",
    )
    destination.add_argument(
        "--out-dir", help="Write one indexed output file per source (required for batches)."
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


def _batch_name(source: str, index: int, fmt: str) -> str:
    stem = Path(source.split("?", 1)[0]).stem or "transcript"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "transcript"
    return f"{index:03d}-{safe}.{fmt}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.source) > 1 and not args.out_dir:
        parser.error("multiple sources require --out-dir")
    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1")
    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1")
    if (args.min_speakers is not None and args.max_speakers is not None
            and args.min_speakers > args.max_speakers):
        parser.error("--min-speakers cannot exceed --max-speakers")
    if not args.diarize and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("speaker-count hints cannot be used with --no-diarize")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        engine = (
            TranscriptionEngine(
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                batch_size=args.batch_size,
                hf_token=args.hf_token,
            )
            if len(args.source) > 1 else None
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    if out_dir:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if args.verbose:
                raise
            print(f"Error: could not create output directory {out_dir}: {exc}", file=sys.stderr)
            return 1

    for index, source in enumerate(args.source, start=1):
        try:
            result = transcribe(
                source,
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
                engine=engine,
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
        output_path = (
            Path(args.output).expanduser() if args.output else
            out_dir / _batch_name(source, index, args.format) if out_dir else None
        )
        if output_path:
            try:
                output_path.write_text(output, encoding="utf-8")
            except OSError as exc:
                if args.verbose:
                    raise
                print(f"Error: could not write {output_path}: {exc}", file=sys.stderr)
                return 1
            speakers = f" | speakers: {', '.join(result.speakers)}" if result.speakers else ""
            print(
                f"Wrote {args.format} -> {output_path} "
                f"({len(result.segments)} segments{speakers})",
                file=sys.stderr,
            )
        else:
            try:
                sys.stdout.write(output)
            except OSError as exc:
                if args.verbose:
                    raise
                print(f"Error: could not write output: {exc}", file=sys.stderr)
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
