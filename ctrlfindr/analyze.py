"""Analysis step: ingested sentences + framework -> coded sentences, five TSVs, codebook."""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import __version__
from .codebook import write_codebook
from .common import (
    DOCUMENTS,
    RESULTS,
    SENTENCES,
    CtrlFindrError,
    ValidationError,
    connect,
    read_table,
    table_exists,
    write_table,
)
from .config import Config
from .engine import analyze_sentences, run_tests
from .framework import Framework, load_framework

log = logging.getLogger("ctrlfindr")

OUTPUT_FILES = (
    "results_document",
    "results_document_clipped",
    "results_document_percentage_of_sentences",
    "results_document_sentiment",
    "results_sentences",
)


def document_outputs(results: pd.DataFrame, fw: Framework) -> dict[str, pd.DataFrame]:
    """The four document-level tables."""
    v = [var.column for var in fw.variables]
    grouped = results.groupby("document", sort=True)
    sums = grouped[v].sum()
    out = {
        "results_document": sums.astype(int).reset_index(),
        "results_document_clipped": sums.clip(upper=1).reset_index(),
    }
    counts = grouped["sentences"].count()
    out["results_document_percentage_of_sentences"] = sums.div(counts, axis=0).multiply(100).reset_index()
    if fw.settings.sentiment:
        # Mean compound score of the sentences flagged by each variable; empty (NaN) when a
        # variable never fired in a document. Documents in order of first appearance.
        docs = results["document"].drop_duplicates()
        table = pd.DataFrame({"document": docs.to_numpy()})
        for col in v:
            means = results.loc[results[col], ["document", "vader_compound"]].groupby("document")["vader_compound"].mean()
            table[col] = docs.map(means).to_numpy()
        out["results_document_sentiment"] = table
    return out


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")


def check_framework(fw: Framework) -> list[str]:
    """Run the framework's tests; raise if they fail and the framework requires them to pass."""
    failures = run_tests(fw)
    if failures and fw.settings.tests_must_pass:
        raise ValidationError(f"{len(failures)} framework test(s) failed (tests sheet)", failures)
    for f in failures:
        log.warning("%s", f)
    if fw.tests and not failures:
        log.info("Framework tests: all %d passed", len(fw.tests))
    return failures


def _package_versions() -> dict[str, str]:
    out = {}
    for pkg in ("pandas", "numpy", "pysbd", "vaderSentiment", "pyahocorasick", "openpyxl", "tqdm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run(cfg: Config, show_progress: bool = True, run_tests_first: bool = True) -> dict[str, pd.DataFrame]:
    """Validate and test the framework, analyze the ingested sentences, write all outputs.

    With ``run_tests_first=False`` failing tests are only counted for run_metadata.json,
    not reported or enforced (the command line has already done that).
    """
    fw = load_framework(cfg.paths.framework)
    log.info("Framework OK: %d lists, %d queries, %d variables, %d tests",
             len(fw.lists), len(fw.queries), len(fw.variables), len(fw.tests))
    failures = check_framework(fw) if run_tests_first else run_tests(fw)

    db = cfg.paths.database
    if not db.is_file():
        raise CtrlFindrError(f"database {db} not found; run the ingest step first")
    with connect(db) as conn:
        if not (table_exists(conn, SENTENCES) and table_exists(conn, DOCUMENTS)):
            raise CtrlFindrError(f"database {db} has no ingested documents; run the ingest step first")
        sentences = read_table(conn, SENTENCES, order_by="document, sentence_id")
        documents = read_table(conn, DOCUMENTS, order_by="document")[["document", "path", "sha256"]]
    if sentences.empty:
        raise CtrlFindrError("the database contains no sentences; check the text files and run ingest again")
    log.info("Analyzing %d sentences from %d documents", len(sentences), sentences["document"].nunique())

    results = analyze_sentences(sentences, fw, long_segment_words=cfg.ingest.long_segment_words,
                                show_progress=show_progress)
    out_dir = cfg.paths.output
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    if cfg.export.document_level:
        outputs.update(document_outputs(results, fw))
    if cfg.export.sentence_level:
        outputs["results_sentences"] = results
    for name, df in outputs.items():
        write_tsv(df, out_dir / f"{name}.tsv")
    # Remove result files of an earlier run that this run did not produce, so the folder
    # never mixes results of different runs.
    stale = [p for p in (out_dir / f"{name}.tsv" for name in OUTPUT_FILES if name not in outputs) if p.is_file()]
    for p in stale:
        p.unlink()
    if stale:
        log.info("Removed %s from an earlier run (not produced with the current settings)",
                 ", ".join(p.name for p in stale))
    with connect(db) as conn:
        write_table(conn, results, RESULTS)
    written = [f"{name}.tsv" for name in outputs]
    write_codebook(fw, out_dir / "codebook.md")
    written.append("codebook.md")

    validation = None
    validation_files = [out_dir / "validation_report.tsv", out_dir / "validation_disagreements.tsv"]
    if cfg.paths.validation.is_file():
        from .validation import validation_report

        validation = validation_report(cfg.paths.validation, results, fw, out_dir)
        written += [p.name for p in validation_files]
    else:  # no validation file any more: do not leave an old report behind
        for p in validation_files:
            if p.is_file():
                p.unlink()
                log.info("Removed %s from an earlier run (no validation file now)", p.name)

    meta = {
        "ctrlfindr_version": __version__,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": cfg.to_dict(),
        "framework": {
            "file": str(fw.path),
            "sha256": fw.sha256,
            "meta": fw.meta,
            "settings": asdict(fw.settings),
            "tests": {"total": len(fw.tests), "failed": len(failures)},
        },
        "corpus": {
            "documents": len(documents),
            "sentences": int(len(results)),
            "files": documents.to_dict(orient="records"),
        },
        "validation": validation,
        "outputs": sorted(written),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    log.info("Wrote %s to %s", ", ".join(sorted(written) + ["run_metadata.json"]), out_dir)
    return outputs
