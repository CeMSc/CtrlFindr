"""Command line: ``python -m ctrlfindr {ingest,analyze,run} --config path/to/ctrlfindr.toml``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .common import CtrlFindrError, setup_logging
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctrlfindr", description="Sentence-level content analysis of text files.")
    parser.add_argument("--version", action="version", version=f"CtrlFindr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("ingest", "read the text files into the database (only new or changed files are processed)"),
        ("analyze", "apply the assessment framework to the ingested sentences and write the results"),
        ("run", "ingest, then analyze"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", "-c", default="ctrlfindr.toml", help="path to ctrlfindr.toml (default: ./ctrlfindr.toml)")
        p.add_argument("--verbose", "-v", action="store_true", help="show debug messages")
        p.add_argument("--no-progress", action="store_true", help="hide progress bars")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        setup_logging(logging.DEBUG if args.verbose else logging.INFO, cfg.paths.output / "ctrlfindr.log")
        progress = not args.no_progress
        if args.command in ("analyze", "run"):
            # Validate the framework before spending time on ingestion.
            from .framework import load_framework

            load_framework(cfg.paths.framework, sentiment_enabled=cfg.sentiment.enabled)
        if args.command in ("ingest", "run"):
            from . import ingest

            ingest.run(cfg, show_progress=progress)
        if args.command in ("analyze", "run"):
            from . import analyze

            analyze.run(cfg, show_progress=progress)
    except CtrlFindrError as exc:
        log = logging.getLogger("ctrlfindr")
        if log.handlers:
            log.error("%s", exc)
        else:  # the config could not be loaded, so logging is not set up yet
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
