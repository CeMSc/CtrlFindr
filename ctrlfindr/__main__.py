"""Command line: ``python -m ctrlfindr COMMAND --config path/to/ctrlfindr.toml``.

Commands: template (blank framework), check (framework only), ingest, analyze,
run (ingest + analyze), sample.
"""

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
    commands = {
        "check": "validate the assessment framework and run its tests (no text needed)",
        "ingest": "read the text files into the database (only new or changed files are processed)",
        "analyze": "apply the framework to the ingested sentences and write the results",
        "run": "ingest, then analyze",
        "sample": "write a blind sample of sentences to hand-code for validation",
    }
    tp = sub.add_parser("template", help="write a blank assessment framework (.ods or .xlsx)")
    tp.add_argument("path", help="file to create, e.g. Input/Assessment_framework.xlsx")
    for name, help_ in commands.items():
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", "-c", default="ctrlfindr.toml", help="path to ctrlfindr.toml (default: ./ctrlfindr.toml)")
        p.add_argument("--verbose", "-v", action="store_true", help="show debug messages")
        p.add_argument("--no-progress", action="store_true", help="hide progress bars")
        if name == "sample":
            p.add_argument("--n", type=int, default=50, help="sentences per stratum (flagged / not flagged) and variable")
            p.add_argument("--variables", default="", help="comma-separated variable numbers (default: all)")
            p.add_argument("--seed", type=int, default=1, help="random seed, recorded for reproducibility")
    args = parser.parse_args(argv)

    log = logging.getLogger("ctrlfindr")
    if args.command == "template":
        from pathlib import Path

        from .template import write_template

        path = Path(args.path)
        if path.exists():
            print(f"ERROR: {path} already exists; choose another file name", file=sys.stderr)
            return 2
        if path.suffix.lower() not in (".ods", ".xlsx"):
            print("ERROR: the file name must end in .ods or .xlsx", file=sys.stderr)
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        write_template(path)
        print(f"Wrote {path}")
        return 0
    try:
        cfg = load_config(args.config)
        setup_logging(logging.DEBUG if args.verbose else logging.INFO, cfg.paths.output / "ctrlfindr.log")
        progress = not args.no_progress
        if args.command in ("check", "analyze", "run"):
            # Validate and test the framework before spending time on ingestion.
            from .analyze import check_framework
            from .framework import load_framework

            fw = load_framework(cfg.paths.framework)
            log.info("Framework %s is valid: %d lists, %d queries, %d variables",
                     cfg.paths.framework.name, len(fw.lists), len(fw.queries), len(fw.variables))
            check_framework(fw)
        if args.command in ("ingest", "run"):
            from . import ingest

            ingest.run(cfg, show_progress=progress)
        if args.command in ("analyze", "run"):
            from . import analyze

            analyze.run(cfg, show_progress=progress, run_tests_first=False)
        if args.command == "sample":
            from .validation import sample

            targets = [t.strip() for t in args.variables.split(",") if t.strip()] or None
            sample(cfg, n=args.n, targets=targets, seed=args.seed)
    except KeyboardInterrupt:
        print("Stopped. Run the same command again to continue: finished files are kept.", file=sys.stderr)
        return 130
    except CtrlFindrError as exc:
        if log.handlers:
            log.error("%s", exc)
        else:  # the config could not be loaded, so logging is not set up yet
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
