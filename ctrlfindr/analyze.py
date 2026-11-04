"""Analysis step: sentences in SQLite + framework -> coded sentences and the five TSVs."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import query as q
from .common import (
    DOCUMENTS,
    RESULTS,
    SENTENCES,
    CtrlFindrError,
    connect,
    normalize_text,
    read_table,
    table_exists,
    write_table,
)
from .config import Config, SentimentSettings
from .framework import SENTIMENT_TERMS, Framework, load_framework
from .matching import Matcher, co_occurrence_column

log = logging.getLogger("ctrlfindr")

OUTPUT_FILES = (
    "results_document",
    "results_document_clipped",
    "results_document_percentage_of_sentences",
    "results_document_sentiment",
    "results_sentences",
)


def _progress(desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        return None
    return lambda it: tqdm(it, desc=desc, unit=" sentences", leave=False, mininterval=0.5)


def sentiment_flags(compound: np.ndarray, settings: SentimentSettings) -> dict[str, np.ndarray]:
    """The five sentiment flags for VADER compound scores.

    With thresholds +-t: positive >= t, negative <= -t, neutral strictly in between.
    A score of exactly 0 is always neutral, so thresholds of 0 reproduce CtrlFindr 2.x
    (positive > 0, neutral == 0, negative < 0).
    """
    c = np.asarray(compound, dtype=float)
    neutral = ((c > settings.negative_threshold) & (c < settings.positive_threshold)) | (c == 0)
    positive = (c >= settings.positive_threshold) & ~neutral
    negative = (c <= settings.negative_threshold) & ~neutral
    return {
        "sentiment_positive": positive,
        "sentiment_neutral": neutral,
        "sentiment_neutralpositive": positive | neutral,
        "sentiment_negative": negative,
        "sentiment_neutralnegative": negative | neutral,
    }


def analyze_sentences(
    sentences: pd.DataFrame,
    fw: Framework,
    sentiment: SentimentSettings | None,
    long_segment_words: int = 500,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Code every sentence. ``sentences`` needs ``document`` and ``text`` columns, plus
    ``vader_compound`` when ``sentiment`` is given; ``sentence_id``/``start``/``end`` are
    carried through when present. Returns one row per sentence with all result columns.
    """
    sentences = sentences.reset_index(drop=True)
    text = [normalize_text(t).lower() for t in sentences["text"].fillna("").astype(str)]
    n = len(text)
    cols: dict[str, np.ndarray] = {}

    matcher = Matcher(fw.taxonomy)
    span_lists = {c.first for c in fw.co_occurrences} | {c.second for c in fw.co_occurrences}
    hits, spans = matcher.match(text, span_lists, _progress("matching") if show_progress else None)
    cols.update(hits)

    token_cache: dict = {}
    for co in fw.co_occurrences:
        cols[co.name] = co_occurrence_column(text, spans[co.first], spans[co.second], co.distance, token_cache)

    doc_codes = pd.factorize(sentences["document"])[0]
    for dc in fw.doc_conditionals:
        cols[dc.name] = pd.Series(cols[dc.list]).groupby(doc_codes).transform("any").to_numpy(bool)

    numeric: dict[str, np.ndarray] = {}
    if sentiment is not None:
        compound = pd.to_numeric(sentences["vader_compound"], errors="coerce").to_numpy(float)
        if np.isnan(compound).any():
            raise CtrlFindrError("some sentences have no sentiment score; run the ingest step again")
        numeric["vader_compound"] = compound
        cols.update(sentiment_flags(compound, sentiment))

    by_name = {qq.name: qq for qq in fw.queries}
    for name in fw.query_order:
        cols[name] = q.evaluate(by_name[name].ast, cols)

    var_cols = {}
    for v in fw.variables:
        var_cols[v.column] = np.logical_or.reduce([cols[c] for c in v.children]) if v.children else np.zeros(n, bool)

    if long_segment_words and n:
        lengths = np.fromiter((len(t.split()) for t in text), dtype=int, count=n)
        long_rows = np.flatnonzero(lengths > long_segment_words)
        if len(long_rows):
            examples = ", ".join(
                f"{sentences['document'].iat[i]} (sentence {sentences['sentence_id'].iat[i] if 'sentence_id' in sentences else i + 1}, {lengths[i]} words)"
                for i in long_rows[:3]
            )
            log.warning(
                "%d segment(s) are longer than %d words, which usually means sentence splitting "
                "failed (tables, reference lists, text without punctuation). They are slow to "
                "process and distort per-sentence percentages. First: %s",
                len(long_rows), long_segment_words, examples,
            )

    head = [c for c in ("sentence_id", "start", "end") if c in sentences.columns]
    parts: dict[str, object] = {c: sentences[c].to_numpy() for c in head}
    parts["document"] = sentences["document"].to_numpy()
    parts["sentences"] = np.array(text, dtype=object)
    order = (
        list(fw.taxonomy)
        + [c.name for c in fw.co_occurrences]
        + [d.name for d in fw.doc_conditionals]
    )
    for name in order:
        parts[name] = cols[name]
    if sentiment is not None:
        parts["vader_compound"] = numeric["vader_compound"]
        for name in SENTIMENT_TERMS:
            parts[name] = cols[name]
    for qq in fw.queries:
        parts[qq.name] = cols[qq.name]
    parts.update(var_cols)
    return pd.DataFrame(parts)


def document_outputs(results: pd.DataFrame, fw: Framework, sentiment: bool) -> dict[str, pd.DataFrame]:
    """The four document-level tables, as in CtrlFindr 2.x."""
    v = [var.column for var in fw.variables]
    grouped = results.groupby("document", sort=True)
    sums = grouped[v].sum()
    out = {
        "results_document": sums.astype(int).reset_index(),
        "results_document_clipped": sums.clip(upper=1).reset_index(),
    }
    counts = grouped["sentences"].count()
    out["results_document_percentage_of_sentences"] = sums.div(counts, axis=0).multiply(100).reset_index()
    if sentiment:
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


def _package_versions() -> dict[str, str]:
    out = {}
    for pkg in ("pandas", "numpy", "pysbd", "vaderSentiment", "pyahocorasick", "tqdm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run(cfg: Config, show_progress: bool = True) -> dict[str, pd.DataFrame]:
    """Validate the framework, analyze the ingested sentences, write all outputs."""
    fw = load_framework(cfg.paths.framework, sentiment_enabled=cfg.sentiment.enabled)
    log.info(
        "Framework OK: %d lists, %d co-occurrences, %d document conditionals, %d queries, %d variables",
        len(fw.taxonomy), len(fw.co_occurrences), len(fw.doc_conditionals), len(fw.queries), len(fw.variables),
    )
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

    results = analyze_sentences(
        sentences,
        fw,
        cfg.sentiment if cfg.sentiment.enabled else None,
        long_segment_words=cfg.ingest.long_segment_words,
        show_progress=show_progress,
    )
    out_dir = cfg.paths.output
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    if cfg.export.document_level:
        outputs.update(document_outputs(results, fw, cfg.sentiment.enabled))
    if cfg.export.sentence_level:
        outputs["results_sentences"] = results
    for name, df in outputs.items():
        write_tsv(df, out_dir / f"{name}.tsv")
    # Remove result files of an earlier run that this run did not produce (an export or
    # sentiment was switched off), so the folder never mixes results of different runs.
    stale = [out_dir / f"{name}.tsv" for name in OUTPUT_FILES if name not in outputs]
    stale = [p for p in stale if p.is_file()]
    for p in stale:
        p.unlink()
    if stale:
        log.info("Removed %s from an earlier run (not produced with the current settings)",
                 ", ".join(p.name for p in stale))
    with connect(db) as conn:
        write_table(conn, results, RESULTS)

    meta = {
        "ctrlfindr_version": __version__,
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": cfg.to_dict(),
        "framework": {"file": str(fw.path), "sha256": fw.sha256},
        "corpus": {
            "documents": len(documents),
            "sentences": int(len(results)),
            "files": documents.to_dict(orient="records"),
        },
        "outputs": sorted(f"{name}.tsv" for name in outputs),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    log.info("Wrote %s to %s", ", ".join(meta["outputs"] + ["run_metadata.json"]), out_dir)
    return outputs
