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
    RESULTS_NEW,
    SENTENCES,
    CtrlFindrError,
    ValidationError,
    connect,
    table_exists,
)
from .config import Config
from .engine import AnalysisStats, analyze_sentences, exclude_sentences, run_tests
from .framework import Framework, load_framework
from .matching import Matcher

log = logging.getLogger("ctrlfindr")

OUTPUT_FILES = (
    "results_document",
    "results_document_clipped",
    "results_document_percentage_of_sentences",
    "results_document_sentiment",
    "results_sentences",
)


class DocumentTables:
    """Builds the four document-level tables from batches of results (whole documents)."""

    def __init__(self, fw: Framework):
        self.fw = fw
        self.columns = [var.column for var in fw.variables]
        self.sums: list[pd.DataFrame] = []
        self.counts: list[pd.Series] = []
        self.sent_sum: list[pd.DataFrame] = []
        self.sent_n: list[pd.DataFrame] = []

    def add(self, results: pd.DataFrame) -> None:
        v = self.columns
        grouped = results.groupby("document", sort=True)
        self.sums.append(grouped[v].sum())
        self.counts.append(grouped["sentences"].count())
        if self.fw.settings.sentiment:
            flags = results[v].to_numpy(bool)
            compound = results["vader_compound"].to_numpy(float)[:, None]
            frame = pd.DataFrame(flags * compound, columns=v)
            frame["document"] = results["document"].to_numpy()
            self.sent_sum.append(frame.groupby("document", sort=True)[v].sum())
            n = pd.DataFrame(flags.astype(int), columns=v)
            n["document"] = results["document"].to_numpy()
            self.sent_n.append(n.groupby("document", sort=True)[v].sum())

    def tables(self) -> dict[str, pd.DataFrame]:
        v = self.columns
        sums = pd.concat(self.sums) if self.sums else pd.DataFrame(columns=v)
        counts = pd.concat(self.counts) if self.counts else pd.Series(dtype=int)
        sums.index.name = counts.index.name = "document"
        out = {
            "results_document": sums.astype(int).reset_index(),
            "results_document_clipped": sums.clip(upper=1).reset_index(),
            "results_document_percentage_of_sentences": sums.div(counts, axis=0).multiply(100).reset_index(),
        }
        if self.fw.settings.sentiment:
            # Mean compound score of the sentences flagged by each variable; empty (NaN) when
            # a variable never fired in a document.
            total, n = pd.concat(self.sent_sum), pd.concat(self.sent_n)
            means = total.div(n.where(n > 0))
            means.index.name = "document"
            out["results_document_sentiment"] = means.reset_index()
        return out


def document_outputs(results: pd.DataFrame, fw: Framework) -> dict[str, pd.DataFrame]:
    """The four document-level tables for a complete results table."""
    acc = DocumentTables(fw)
    acc.add(results)
    return acc.tables()


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
    for pkg in ("pandas", "numpy", "pysbd", "vaderSentiment", "pyahocorasick", "openpyxl", "tqdm"):  # vader optional
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run(cfg: Config, show_progress: bool = True, run_tests_first: bool = True) -> dict[str, pd.DataFrame]:
    """Validate and test the framework, analyze the ingested sentences, write all outputs.

    The sentences are processed in batches of whole documents ([analysis] batch_sentences),
    so memory use does not grow with the corpus. Earlier outputs are only replaced when
    the run completes; a stopped run leaves them as they were. Returns the document tables.

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
    out_dir = cfg.paths.output
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "results_sentences.tsv.partial"
    tables = DocumentTables(fw)
    stats = AnalysisStats()
    collector = None
    if cfg.paths.validation.is_file():
        from .validation import ValidationCollector

        collector = ValidationCollector(cfg.paths.validation, fw)
    matcher = Matcher(fw.all_lists)
    n_sentences = n_scored = 0

    with connect(db) as conn:
        if not (table_exists(conn, SENTENCES) and table_exists(conn, DOCUMENTS)):
            raise CtrlFindrError(f"database {db} has no ingested documents; run the ingest step first")
        documents = pd.read_sql_query(f"SELECT document, path, format, sha256 FROM {DOCUMENTS} ORDER BY document", conn)
        counts = conn.execute(f"SELECT document, COUNT(*) FROM {SENTENCES} GROUP BY document ORDER BY document").fetchall()
        if not counts:
            raise CtrlFindrError("the database contains no sentences; check the text files and run ingest again")
        batches = _batches(counts, cfg.analysis.batch_sentences)
        total = sum(c for _, c in counts)
        log.info("Analyzing %d sentences from %d documents in %d batch(es)", total, len(counts), len(batches))
        conn.execute(f"DROP TABLE IF EXISTS {RESULTS_NEW}")
        bar = None
        if show_progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=total, desc="analyze", unit=" sentences", leave=False, mininterval=0.5)
        try:
            for first, last, n in batches:
                sentences = pd.read_sql_query(
                    f"SELECT * FROM {SENTENCES} WHERE document BETWEEN ? AND ? ORDER BY document, sentence_id",
                    conn, params=(first, last))
                sentences = exclude_sentences(sentences, fw.settings, stats)
                if not sentences.empty:
                    if fw.settings.sentiment:
                        from .sentiment import cached_scores

                        sentences["vader_compound"], scored = cached_scores(conn, sentences)
                        n_scored += scored
                    results = analyze_sentences(sentences, fw, long_segment_words=cfg.ingest.long_segment_words,
                                                matcher=matcher, stats=stats)
                    tables.add(results)
                    if collector is not None:
                        collector.add(results)
                    if cfg.export.sentence_level:
                        results.to_csv(partial, sep="\t", index=False, encoding="utf-8", lineterminator="\n",
                                       mode="w" if n_sentences == 0 else "a", header=n_sentences == 0)
                    results.to_sql(RESULTS_NEW, conn, if_exists="append", index=False)
                    n_sentences += len(results)
                conn.commit()  # sentiment scores of finished batches are kept even if the run stops
                if bar is not None:
                    bar.update(n)
            if n_sentences == 0:
                raise CtrlFindrError("every sentence is excluded by exclude_sentence_types in the settings sheet")
            # Replace the previous results only now that the run is complete, in one transaction
            # (sqlite3 would otherwise run each of these statements on its own).
            conn.execute("BEGIN")
            conn.execute(f"DROP TABLE IF EXISTS {RESULTS}")
            conn.execute(f"ALTER TABLE {RESULTS_NEW} RENAME TO {RESULTS}")
            conn.commit()
        except BaseException:
            conn.rollback()
            conn.execute(f"DROP TABLE IF EXISTS {RESULTS_NEW}")
            conn.commit()
            partial.unlink(missing_ok=True)
            raise
        finally:
            if bar is not None:
                bar.close()
    stats.report()
    if n_scored:
        log.info("Sentiment: %d sentence(s) scored with VADER, the others came from the cache", n_scored)

    outputs: dict[str, pd.DataFrame] = tables.tables() if cfg.export.document_level else {}
    written = []
    for name, df in outputs.items():
        write_tsv(df, out_dir / f"{name}.tsv")
        written.append(f"{name}.tsv")
    if cfg.export.sentence_level:
        partial.replace(out_dir / "results_sentences.tsv")
        written.append("results_sentences.tsv")
    # Remove result files of an earlier run that this run did not produce, so the folder
    # never mixes results of different runs.
    stale = [p for p in (out_dir / f"{name}.tsv" for name in OUTPUT_FILES) if p.is_file() and p.name not in written]
    for p in stale:
        p.unlink()
    if stale:
        log.info("Removed %s from an earlier run (not produced with the current settings)",
                 ", ".join(p.name for p in stale))
    write_codebook(fw, out_dir / "codebook.md")
    written.append("codebook.md")

    validation = None
    validation_files = [out_dir / "validation_report.tsv", out_dir / "validation_disagreements.tsv"]
    if collector is not None:
        validation = collector.finish(out_dir)
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
            "sentences": n_sentences,
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


def _batches(counts: list[tuple[str, int]], size: int) -> list[tuple[str, str, int]]:
    """Group consecutive documents into batches of about ``size`` sentences.

    Returns (first document, last document, sentences) per batch. A document is never split:
    DOC, DOC_SHARE and WINDOW need all of its sentences.
    """
    out: list[tuple[str, str, int]] = []
    current: list[str] = []
    n = 0
    for doc, c in counts:
        if current and n + c > size:
            out.append((current[0], current[-1], n))
            current, n = [], 0
        current.append(doc)
        n += c
    if current:
        out.append((current[0], current[-1], n))
    return out
