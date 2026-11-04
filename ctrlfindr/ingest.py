"""Ingest step: text files -> cleaned documents and scored sentences in SQLite.

Re-running ingest only processes files that are new or changed (by SHA-256) and removes
documents whose files are gone. Changing the cleaning setting, or upgrading CtrlFindr,
pySBD or vaderSentiment, rebuilds everything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from . import __version__
from .common import DOCUMENTS, META, SENTENCES, CtrlFindrError, ValidationError, connect, normalize_text, sha256_file
from .config import Config

log = logging.getLogger("ctrlfindr")

# Bump when cleaning or segmentation logic changes, so existing databases are rebuilt.
INGEST_REVISION = 3

_URL = re.compile(r"http\S+|www\S+|https\S+")
_ELLIPSIS = re.compile(r"\.{2,}")
_PARAGRAPH_BREAK = re.compile(r"[^\S\n]*\n(?:[^\S\n]*\n)+[^\S\n]*")  # a blank line
_LINE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")  # a single line break (not a blank line)
_SPACES = re.compile(r"[^\S\n]+")  # runs of whitespace other than line breaks
_CITATION_AFTER_STOP = re.compile(r"(?<=[.!?])(?=\[\d)")  # "2050.[4] The" -> "2050. [4] The"

_SCHEMA = (
    f"""CREATE TABLE IF NOT EXISTS {DOCUMENTS} (
        document TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, clean_text TEXT NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS {SENTENCES} (
        document TEXT NOT NULL, sentence_id INTEGER NOT NULL, paragraph INTEGER NOT NULL,
        start INTEGER NOT NULL, "end" INTEGER NOT NULL,
        text TEXT NOT NULL, vader_compound REAL NOT NULL, PRIMARY KEY (document, sentence_id))""",
    f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)


@dataclass(frozen=True)
class SourceFile:
    document: str
    path: Path


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    paragraph: int
    start: int
    end: int
    text: str


def discover(text_dir: Path) -> list[SourceFile]:
    """Sorted list of the .txt files in ``text_dir`` (extension matched case-insensitively)."""
    if not text_dir.is_dir():
        raise CtrlFindrError(f"text files folder not found: {text_dir}")
    files = sorted((p for p in text_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"), key=lambda p: p.name)
    if not files:
        raise CtrlFindrError(f"no .txt files in {text_dir}")
    out = [SourceFile(p.name[: -len(p.suffix)], p) for p in files]
    seen: dict[str, Path] = {}
    dupes = []
    for f in out:
        if f.document in seen:
            dupes.append(f"{seen[f.document].name} and {f.path.name} both give document name {f.document!r}")
        seen[f.document] = f.path
    if dupes:
        raise ValidationError("Text files have clashing names", dupes)
    return out


def decode(data: bytes, errors: str) -> str:
    """UTF-8 (a leading byte-order mark is dropped); ``errors`` is 'strict' or 'replace'."""
    return data.decode("utf-8-sig", errors=errors)


def clean_document(raw: str, text_cleaning: bool = True) -> str:
    """Normalize a document. Sentence offsets refer to the string returned here.

    Always: NFKC normalization and dash/quote folding; a blank line becomes a paragraph
    break (``\\n\\n``); any other run of whitespace, including a single line break, becomes
    one space; a space is inserted between a full stop and a following citation mark
    (``2050.[4] The``), which the sentence splitter would otherwise not split. With ``text_cleaning``: URLs are removed and ``..``/``...`` become ``.``.
    """
    text = normalize_text(raw).replace("\r\n", "\n").replace("\r", "\n")
    if text_cleaning:
        text = _URL.sub("", text)
        text = _ELLIPSIS.sub(".", text)
    text = _CITATION_AFTER_STOP.sub(" ", text)
    text = _PARAGRAPH_BREAK.sub("\n\n", text)
    text = _LINE_BREAK.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


class Segmenter:
    """pySBD sentence splitting, paragraph by paragraph, with character offsets."""

    def __init__(self) -> None:
        import pysbd

        self._seg = pysbd.Segmenter(language="en", clean=False)

    def split(self, text: str) -> list[Sentence]:
        out: list[Sentence] = []
        pos = 0
        for p_no, para in enumerate(text.split("\n\n"), start=1):
            cursor = 0
            for piece in self._seg.segment(para) if para.strip() else ():
                s = piece.strip()
                if not s:
                    continue
                k = para.find(s, cursor)
                if k < 0:  # pySBD altered the text; should not happen with clean=False
                    raise RuntimeError(f"sentence splitter changed the text near: {s[:60]!r}")
                cursor = k + len(s)
                if not any(ch.isalpha() for ch in s):  # numbers, punctuation, "[12]" citation marks
                    continue
                out.append(Sentence(len(out) + 1, p_no, pos + k, pos + k + len(s), s))
            pos += len(para) + 2
        return out


def _signature(cfg: Config) -> str:
    def ver(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "?"

    sig = {
        "ctrlfindr": __version__,
        "ingest_revision": INGEST_REVISION,
        "pysbd": ver("pysbd"),
        "vaderSentiment": ver("vaderSentiment"),
        "text_cleaning": cfg.ingest.text_cleaning,
        "encoding_errors": cfg.ingest.encoding_errors,
    }
    return json.dumps(sig, sort_keys=True)


def _prepare_db(conn: sqlite3.Connection, signature: str) -> dict[str, str]:
    """Create tables; drop everything if the stored data was made with other settings.

    Returns {document: sha256} for documents that can be kept as they are.
    """
    def columns(table: str) -> list[str]:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]

    expected = {DOCUMENTS: ["document", "path", "sha256", "clean_text"],
                SENTENCES: ["document", "sentence_id", "paragraph", "start", "end", "text", "vader_compound"]}
    stored = None
    if columns(META) == ["key", "value"]:
        row = conn.execute(f"SELECT value FROM {META} WHERE key='ingest_signature'").fetchone()
        stored = row[0] if row else None
    rebuild = stored != signature or any(columns(t) not in ([], cols) for t, cols in expected.items())
    if rebuild:
        if stored is not None or columns(DOCUMENTS):
            log.info("Ingest settings or versions changed: rebuilding the database")
        for table in (SENTENCES, DOCUMENTS, META):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    for ddl in _SCHEMA:
        conn.execute(ddl)
    conn.execute(f"INSERT OR REPLACE INTO {META} (key, value) VALUES ('ingest_signature', ?)", (signature,))
    return dict(conn.execute(f"SELECT document, sha256 FROM {DOCUMENTS}").fetchall())


def run(cfg: Config, show_progress: bool = True) -> dict[str, int]:
    """Ingest the text files folder into the database. Returns counts of what was done.

    Files are hashed by streaming them; only new or changed files are read into memory,
    one at a time, so memory use does not grow with the size of the corpus.
    """
    files = {f.document: f for f in discover(cfg.paths.text_files)}
    digests = {d: sha256_file(f.path) for d, f in files.items()}
    errors = cfg.ingest.encoding_errors

    db = cfg.paths.database
    with connect(db) as conn:
        kept = _prepare_db(conn, _signature(cfg))
        removed = [d for d in kept if d not in files]
        todo = [d for d in files if kept.get(d) != digests[d]]

        # Check encodings before anything is changed. Unchanged files were checked when
        # they were ingested (the encoding setting is part of the stored signature).
        bad = []
        for d in todo:
            try:
                decode(files[d].path.read_bytes(), "strict")
            except UnicodeDecodeError as exc:
                bad.append(f"{files[d].path.name}: not valid UTF-8 (byte {exc.start})")
        if bad and errors == "strict":
            raise ValidationError(
                "Some text files are not UTF-8. Re-save them as UTF-8, or set "
                '[ingest] encoding_errors = "replace" to substitute unreadable characters',
                bad,
            )
        if bad:
            log.warning("%d file(s) are not valid UTF-8; unreadable bytes were replaced: %s",
                        len(bad), "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))

        for d in removed + todo:
            conn.execute(f"DELETE FROM {SENTENCES} WHERE document=?", (d,))
            conn.execute(f"DELETE FROM {DOCUMENTS} WHERE document=?", (d,))
        for d, f in files.items():  # files may have moved; keep paths current
            conn.execute(f"UPDATE {DOCUMENTS} SET path=? WHERE document=?", (str(f.path), d))

        n_sent = 0
        if todo:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            segmenter = Segmenter()
            sia = SentimentIntensityAnalyzer()
            iterator = todo
            if show_progress:
                from tqdm.auto import tqdm

                iterator = tqdm(todo, desc="ingest", unit=" files", leave=False)
            empty = []
            for d in iterator:
                f = files[d]
                data = f.path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()  # the bytes actually ingested
                text = clean_document(decode(data, errors), cfg.ingest.text_cleaning)
                del data
                sents = segmenter.split(text)
                if not sents:
                    empty.append(f.path.name)
                conn.execute(
                    f"INSERT INTO {DOCUMENTS} (document, path, sha256, clean_text) VALUES (?, ?, ?, ?)",
                    (d, str(f.path), digest, text),
                )
                conn.executemany(
                    f'INSERT INTO {SENTENCES} (document, sentence_id, paragraph, start, "end", text, vader_compound) '
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(d, s.sentence_id, s.paragraph, s.start, s.end, s.text, sia.polarity_scores(s.text)["compound"])
                     for s in sents],
                )
                n_sent += len(sents)
            if empty:
                log.warning("%d file(s) contain no sentences and will not appear in the results: %s",
                            len(empty), ", ".join(empty))
        total = conn.execute(f"SELECT COUNT(*) FROM {SENTENCES}").fetchone()[0]

    summary = {"files": len(files), "processed": len(todo), "unchanged": len(files) - len(todo),
               "removed": len(removed), "new_sentences": n_sent, "sentences": total}
    log.info("Ingest: %(files)d files (%(processed)d processed, %(unchanged)d unchanged, "
             "%(removed)d removed); %(sentences)d sentences in the database", summary)
    return summary
