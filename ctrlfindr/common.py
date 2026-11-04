"""Shared helpers: errors, text normalization, logging, SQLite access."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

LOGGER_NAME = "ctrlfindr"
log = logging.getLogger(LOGGER_NAME)


class CtrlFindrError(Exception):
    """An error caused by the user's input or settings (reported without a traceback)."""


class ValidationError(CtrlFindrError):
    """One or more problems found while validating inputs; all are reported together."""

    def __init__(self, title: str, problems: list[str]):
        self.title = title
        self.problems = problems
        lines = [f"{title} ({len(problems)} problem{'s' if len(problems) != 1 else ''}):"]
        lines += [f"  - {p}" for p in problems]
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------------- text

# Typographic variants folded to their ASCII form, so that a term typed with a plain
# hyphen or straight quote matches text that uses an en dash or curly quote (and the
# reverse). Invisible characters that would split words are removed.
_FOLD = str.maketrans(
    {
        **{c: "-" for c in "‐‑‒–—―−﹘﹣－"},
        **{c: "'" for c in "‘’‚‛′ʼ"},
        **{c: '"' for c in "“”„‟″«»"},
        "­": None,  # soft hyphen
        "​": None,  # zero-width space
        "‌": None,
        "‍": None,
        "﻿": None,  # byte-order mark / zero-width no-break space
    }
)


def normalize_text(text: str) -> str:
    """NFKC-normalize and fold dashes/quotes. Idempotent; applied to corpus and framework."""
    return unicodedata.normalize("NFKC", text).translate(_FOLD)


_WS = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------ logging

def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure the package logger once (console, plus an optional UTF-8 log file)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    logger.propagate = False


# ------------------------------------------------------------------------- SQLite

DOCUMENTS = "documents"
SENTENCES = "sentences"
RESULTS = "results_sentences"
SENTIMENT = "sentiment"  # cache of VADER scores, filled by the analysis when sentiment is on
META = "meta"
_TABLES = {DOCUMENTS, SENTENCES, RESULTS, SENTIMENT, META}


def _table(name: str) -> str:
    """Return a table name after checking it against the fixed whitelist."""
    if name not in _TABLES:
        raise ValueError(f"unknown table {name!r}")
    return name


@contextlib.contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database, commit on success, roll back on error, always close."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_table(name),)
    ).fetchone()
    return row is not None


def read_table(conn: sqlite3.Connection, name: str, order_by: str = "") -> pd.DataFrame:
    sql = f'SELECT * FROM "{_table(name)}"'
    if order_by:
        if not re.fullmatch(r"[a-z_]+(, [a-z_]+)*", order_by):
            raise ValueError(f"invalid ORDER BY {order_by!r}")
        sql += f" ORDER BY {order_by}"
    return pd.read_sql_query(sql, conn)


def write_table(conn: sqlite3.Connection, df: pd.DataFrame, name: str) -> None:
    df.to_sql(_table(name), conn, if_exists="replace", index=False)
