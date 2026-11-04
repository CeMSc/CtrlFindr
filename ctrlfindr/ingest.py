"""Ingest step: text files -> cleaned documents and sentences in SQLite.

Two input types are read from the text files folder:

* ``.txt`` files are plain, unstructured text. Every sentence gets the structure label
  ``unstructured``.
* ``.md`` files are Markdown. Headings, list items, table rows, footnotes and body text are
  told apart (see ctrlfindr.markdown), Markdown syntax is removed, and every sentence gets
  the label ``header``, ``list``, ``table``, ``footnote`` or ``body``.

Re-running ingest only processes files that are new or changed (by SHA-256) and removes
documents whose files are gone. Changing the cleaning setting, or upgrading CtrlFindr or
pySBD, rebuilds everything. Sentiment is not computed here (see ctrlfindr.sentiment).

Each file is saved in its own transaction as soon as it is done, so an ingest that is
stopped (Ctrl+C, a killed process, a crash) keeps every finished file; running it again
continues with the files that are left.
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
from .common import (DOCUMENTS, META, SENTENCES, SENTIMENT, CtrlFindrError, ValidationError, connect,
                     normalize_text, sha256_file)
from .config import Config
from .markdown import blocks as markdown_blocks

log = logging.getLogger("ctrlfindr")

# Bump when cleaning or segmentation logic changes, so existing databases are rebuilt.
INGEST_REVISION = 6
SUFFIXES = {".txt": "txt", ".md": "md"}
UNSTRUCTURED = "unstructured"
STRUCTURE_LABELS = ("header", "body", "list", "table", "footnote", UNSTRUCTURED)
_SINGLE_SEGMENT = {"header", "table"}  # never split into several sentences

_URL = re.compile(r"http\S+|www\S+|https\S+")
_ELLIPSIS = re.compile(r"\.{2,}")
_BLANK_LINE = re.compile(r"[^\S\n]*\n(?:[^\S\n]*\n)+[^\S\n]*")  # one or more blank lines
_LINE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")  # a single line break (not a blank line)
_SPACES = re.compile(r"[^\S\n]+")  # runs of whitespace other than line breaks
_CITATION_AFTER_STOP = re.compile(r"(?<=[.!?])(?=\[\d)")  # "2050.[4] The" -> "2050. [4] The"
# "published.The minister" -> "published. The minister": a full stop (or ? !, possibly with a
# closing bracket or quote) after two lower-case letters or digits, directly followed by a
# capitalised word. "U.S.Army", "Ph.D.Later", "St.Louis", "e.g.The" and "4.2.Results" are left alone.
_MISSING_SPACE = re.compile(r"(?<=[a-z0-9]{2})([)\]\"']?[.!?][)\]\"']?)(?=[A-Z][a-z])")

_SCHEMA = (
    f"""CREATE TABLE IF NOT EXISTS {DOCUMENTS} (
        document TEXT PRIMARY KEY, path TEXT NOT NULL, format TEXT NOT NULL, sha256 TEXT NOT NULL,
        clean_text TEXT NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS {SENTENCES} (
        document TEXT NOT NULL, sentence_id INTEGER NOT NULL, structure TEXT NOT NULL,
        start INTEGER NOT NULL, "end" INTEGER NOT NULL, text TEXT NOT NULL,
        PRIMARY KEY (document, sentence_id))""",
    f"CREATE TABLE IF NOT EXISTS {META} (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
_COLUMNS = {DOCUMENTS: ["document", "path", "format", "sha256", "clean_text"],
            SENTENCES: ["document", "sentence_id", "structure", "start", "end", "text"]}


@dataclass(frozen=True)
class SourceFile:
    document: str
    path: Path

    @property
    def format(self) -> str:
        return SUFFIXES[self.path.suffix.lower()]


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    structure: str
    start: int
    end: int
    text: str


def discover(text_dir: Path) -> list[SourceFile]:
    """Sorted list of the .txt and .md files in ``text_dir`` (extensions in any case)."""
    if not text_dir.is_dir():
        raise CtrlFindrError(f"text files folder not found: {text_dir}")
    files = sorted((p for p in text_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
                   key=lambda p: p.name)
    if not files:
        raise CtrlFindrError(f"no .txt or .md files in {text_dir}")
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


def clean_document(raw: str, text_cleaning: bool = True, repair_spacing: bool = False) -> str:
    """Normalize plain text. Sentence offsets refer to the string returned here.

    Always: NFKC normalization and dash/quote folding; blank lines become ``\\n\\n``, where a
    sentence always ends; any other run of whitespace, including a single line break, becomes
    one space; a space is inserted between a full stop and a following citation mark
    (``2050.[4] The``), which the sentence splitter would otherwise not split.
    With ``text_cleaning``: URLs are removed and ``..``/``...`` become ``.``.
    With ``repair_spacing`` (used for .txt files): a missing space after a full stop is
    added (``published.The minister`` -> ``published. The minister``).
    """
    text = normalize_text(raw).replace("\r\n", "\n").replace("\r", "\n")
    if text_cleaning:
        text = _URL.sub("", text)
        text = _ELLIPSIS.sub(".", text)
    text = _CITATION_AFTER_STOP.sub(" ", text)
    if repair_spacing:
        text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _BLANK_LINE.sub("\n\n", text)
    text = _LINE_BREAK.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def prepare(raw: str, fmt: str, text_cleaning: bool = True) -> tuple[str, list[str]]:
    """Cleaned text of a document and the structure label of each of its blocks.

    Blocks are separated by ``\\n\\n`` in the cleaned text. For Markdown, each heading, list
    item, table row and body paragraph is one block, with the Markdown syntax removed.
    """
    if fmt == "txt":
        text = clean_document(raw, text_cleaning, repair_spacing=True)
        return text, [UNSTRUCTURED] * (text.count("\n\n") + 1)
    parts, labels = [], []
    for label, block in markdown_blocks(raw):
        cleaned = clean_document(block, text_cleaning).replace("\n\n", " ")
        if cleaned:
            parts.append(cleaned)
            labels.append(label)
    return "\n\n".join(parts), labels


class Segmenter:
    """pySBD sentence splitting with character offsets. A block boundary (blank line) always
    ends a sentence; headings and table rows are kept whole."""

    def __init__(self) -> None:
        import pysbd

        self._seg = pysbd.Segmenter(language="en", clean=False)

    def split(self, text: str, labels: list[str] | None = None) -> list[Sentence]:
        out: list[Sentence] = []
        pos = 0
        for b, block in enumerate(text.split("\n\n")):
            label = labels[b] if labels else UNSTRUCTURED
            if not block.strip():
                pieces = []
            elif label in _SINGLE_SEGMENT:
                pieces = [block]
            else:
                pieces = self._seg.segment(block)
            cursor = 0
            for piece in pieces:
                s = piece.strip()
                if not s:
                    continue
                k = block.find(s, cursor)
                if k < 0:  # pySBD altered the text; should not happen with clean=False
                    raise RuntimeError(f"sentence splitter changed the text near: {s[:60]!r}")
                cursor = k + len(s)
                if not any(ch.isalpha() for ch in s):  # numbers, punctuation, "[12]" citation marks
                    continue
                out.append(Sentence(len(out) + 1, label, pos + k, pos + k + len(s), s))
            pos += len(block) + 2
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

    stored = None
    if columns(META) == ["key", "value"]:
        row = conn.execute(f"SELECT value FROM {META} WHERE key='ingest_signature'").fetchone()
        stored = row[0] if row else None
    rebuild = stored != signature or any(columns(t) not in ([], cols) for t, cols in _COLUMNS.items())
    if rebuild:
        if stored is not None or columns(DOCUMENTS):
            log.info("Ingest settings or versions changed: rebuilding the database")
        for table in (SENTIMENT, SENTENCES, DOCUMENTS, META):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    for ddl in _SCHEMA:
        conn.execute(ddl)
    conn.execute(f"INSERT OR REPLACE INTO {META} (key, value) VALUES ('ingest_signature', ?)", (signature,))
    return dict(conn.execute(f"SELECT document, sha256 FROM {DOCUMENTS}").fetchall())


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


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

        cached_sentiment = _table_exists(conn, SENTIMENT)

        def forget(d: str) -> None:
            conn.execute(f"DELETE FROM {SENTENCES} WHERE document=?", (d,))
            conn.execute(f"DELETE FROM {DOCUMENTS} WHERE document=?", (d,))
            if cached_sentiment:  # scores of changed documents no longer match their sentences
                conn.execute(f"DELETE FROM {SENTIMENT} WHERE document=?", (d,))

        for d in removed:
            forget(d)
        for d, f in files.items():  # files may have moved; keep paths current
            conn.execute(f"UPDATE {DOCUMENTS} SET path=? WHERE document=?", (str(f.path), d))
        conn.commit()  # settings check, removals and paths are saved before the long part
        if todo:
            log.info("Ingest: %d file(s) to process, %d unchanged", len(todo), len(files) - len(todo))

        n_sent = done = 0
        if todo:
            segmenter = Segmenter()
            iterator = todo
            if show_progress:
                from tqdm.auto import tqdm

                iterator = tqdm(todo, desc="ingest", unit=" files", leave=False)
            empty = []
            try:
                for d in iterator:
                    f = files[d]
                    data = f.path.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()  # the bytes actually ingested
                    text, labels = prepare(decode(data, errors), f.format, cfg.ingest.text_cleaning)
                    del data
                    sents = segmenter.split(text, labels)
                    if not sents:
                        empty.append(f.path.name)
                    # One transaction per file: until it is committed, the previous version of
                    # the document (if any) stays in place, so a stop never leaves half a file.
                    forget(d)
                    conn.execute(
                        f"INSERT INTO {DOCUMENTS} (document, path, format, sha256, clean_text) VALUES (?, ?, ?, ?, ?)",
                        (d, str(f.path), f.format, digest, text),
                    )
                    conn.executemany(
                        f'INSERT INTO {SENTENCES} (document, sentence_id, structure, start, "end", text) '
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [(d, s.sentence_id, s.structure, s.start, s.end, s.text) for s in sents],
                    )
                    conn.commit()
                    done += 1
                    n_sent += len(sents)
            except KeyboardInterrupt:
                conn.rollback()
                log.warning("Ingest stopped after %d of %d file(s). The finished files are saved; run ingest "
                            "again to continue with the %d file(s) left.", done, len(todo), len(todo) - done)
                raise
            if empty:
                log.warning("%d file(s) contain no sentences and will not appear in the results: %s",
                            len(empty), ", ".join(empty))
        total = conn.execute(f"SELECT COUNT(*) FROM {SENTENCES}").fetchone()[0]

    summary = {"files": len(files), "processed": len(todo), "unchanged": len(files) - len(todo),
               "removed": len(removed), "new_sentences": n_sent, "sentences": total}
    log.info("Ingest: %(files)d files (%(processed)d processed, %(unchanged)d unchanged, "
             "%(removed)d removed); %(sentences)d sentences in the database", summary)
    return summary
