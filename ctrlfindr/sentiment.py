"""Optional VADER sentiment scores.

Sentiment is used only when a framework sets ``sentiment = true``. The scores are then
computed during the analysis, on the original-case sentence, and cached in the database
table ``sentiment``, so a later analysis only scores sentences it has not seen. Ingest
removes the cached scores of documents it re-processes. The package vaderSentiment is an
optional install (``pip install vaderSentiment``).
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
from importlib import metadata

import numpy as np
import pandas as pd

from .common import META, SENTIMENT, CtrlFindrError

log = logging.getLogger("ctrlfindr")
MISSING = ("sentiment = true needs the vaderSentiment package, which is not installed: "
           "run 'pip install vaderSentiment', or set sentiment = false in the settings sheet")


def available() -> bool:
    return importlib.util.find_spec("vaderSentiment") is not None


def _analyzer():
    if not available():
        raise CtrlFindrError(MISSING)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def version() -> str:
    try:
        return metadata.version("vaderSentiment")
    except metadata.PackageNotFoundError:
        return "?"


def score_texts(texts) -> list[float]:
    """VADER compound score of each text (no caching)."""
    sia = _analyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def cached_scores(conn: sqlite3.Connection, sentences: pd.DataFrame) -> np.ndarray:
    """Compound scores for ``sentences`` (document, sentence_id, text), using and filling the cache."""
    _analyzer()  # fail early with a clear message
    conn.execute(f"CREATE TABLE IF NOT EXISTS {SENTIMENT} (document TEXT NOT NULL, sentence_id INTEGER NOT NULL, "
                 "vader_compound REAL NOT NULL, PRIMARY KEY (document, sentence_id))")
    conn.execute(f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute(f"SELECT value FROM {META} WHERE key='sentiment_version'").fetchone()
    if row is None or row[0] != version():
        conn.execute(f"DELETE FROM {SENTIMENT}")  # scores from another vaderSentiment version
        conn.execute(f"INSERT OR REPLACE INTO {META} (key, value) VALUES ('sentiment_version', ?)", (version(),))
    cache = pd.read_sql_query(f"SELECT document, sentence_id, vader_compound FROM {SENTIMENT}", conn)
    merged = sentences[["document", "sentence_id", "text"]].merge(cache, on=["document", "sentence_id"], how="left")
    missing = merged["vader_compound"].isna().to_numpy()
    if missing.any():
        log.info("Scoring sentiment of %d sentence(s) with VADER", int(missing.sum()))
        new = score_texts(merged.loc[missing, "text"].tolist())
        merged.loc[missing, "vader_compound"] = new
        conn.executemany(
            f"INSERT OR REPLACE INTO {SENTIMENT} (document, sentence_id, vader_compound) VALUES (?, ?, ?)",
            list(zip(merged.loc[missing, "document"], merged.loc[missing, "sentence_id"].astype(int), new)),
        )
    return merged["vader_compound"].to_numpy(float)
