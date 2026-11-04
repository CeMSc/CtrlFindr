"""Evaluate a framework on a table of sentences, and run the framework's own tests."""

from __future__ import annotations

import logging
import operator
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import query as q
from .common import CtrlFindrError, normalize_text
from .framework import SENTIMENT_TERMS, Framework, Settings
from .matching import Match, Matcher, count, count_distinct, near, not_near, word_tokens

log = logging.getLogger("ctrlfindr")

_COMPARE = {">=": operator.ge, ">": operator.gt, "=": operator.eq, "!=": operator.ne,
            "<=": operator.le, "<": operator.lt}


def sentiment_flags(compound: np.ndarray, settings: Settings) -> dict[str, np.ndarray]:
    """The five sentiment flags for VADER compound scores.

    positive >= positive threshold, negative <= negative threshold, neutral in between.
    A score of exactly 0 is always neutral.
    """
    c = np.asarray(compound, dtype=float)
    pos, neg = settings.sentiment_positive_threshold, settings.sentiment_negative_threshold
    neutral = ((c > neg) & (c < pos)) | (c == 0)
    positive = (c >= pos) & ~neutral
    negative = (c <= neg) & ~neutral
    return {
        "sentiment_positive": positive,
        "sentiment_neutral": neutral,
        "sentiment_neutralpositive": positive | neutral,
        "sentiment_negative": negative,
        "sentiment_neutralnegative": negative | neutral,
    }


class Evaluator:
    """Holds the per-sentence columns and evaluates expressions over them."""

    def __init__(self, fw: Framework, text: list[str], documents: np.ndarray):
        self.fw = fw
        self.text = text
        self.n = len(text)
        self.doc = pd.factorize(documents)[0]
        if self.n and np.count_nonzero(np.diff(self.doc)) != self.doc.max():
            # WINDOW and the document functions need the rows of a document to be contiguous
            raise CtrlFindrError("the sentences of each document must be consecutive rows")
        self.columns: dict[str, np.ndarray] = {}
        self.matches: dict[str, list[list[Match] | None]] = {}
        self.variables: dict[str, np.ndarray] = {}
        self._tokens: dict[int, tuple] = {}
        if self.n:
            codes = self.doc
            first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
            last = np.r_[first[1:], self.n] - 1
            run = np.repeat(np.arange(len(first)), np.diff(np.r_[first, self.n]))
            self.doc_first, self.doc_last = first[run], last[run]

    # ------------------------------------------------------------------ helpers

    def tokens(self, row: int):
        t = self._tokens.get(row)
        if t is None:
            t = self._tokens[row] = word_tokens(self.text[row])
        return t

    def set_matches(self, arg: q.Node) -> list[list[Match]]:
        lists = self.fw.set_lists[id(arg)]
        out: list[list[Match]] = [[] for _ in range(self.n)]
        for name in lists:
            for row, m in enumerate(self.matches[name]):
                if m:
                    out[row].extend(m)
        if len(lists) > 1:
            out = [sorted(set(m)) for m in out]
        return out

    def _group_count(self, codes: np.ndarray, inner: np.ndarray) -> np.ndarray:
        if not self.n:
            return np.zeros(0)
        return np.bincount(codes, weights=inner.astype(float))[codes]

    # --------------------------------------------------------------- evaluation

    def evaluate(self, node: q.Node) -> np.ndarray:
        if isinstance(node, q.Name):
            return self.columns[node.name]
        if isinstance(node, q.InlineTerm):
            return self.columns["{" + node.text + "}"]
        if isinstance(node, q.Not):
            return ~self.evaluate(node.operand)
        if isinstance(node, (q.And, q.Or)):
            parts = [self.evaluate(o) for o in node.operands]
            return (np.logical_and if isinstance(node, q.And) else np.logical_or).reduce(parts)
        return self._call(node)

    def _call(self, node: q.Call) -> np.ndarray:
        f = node.func
        out = np.zeros(self.n, dtype=bool)
        if f == "v":
            return self.variables[node.args[0]]
        if f in ("near", "before", "not_near"):
            a, b = self.set_matches(node.args[0]), self.set_matches(node.args[1])
            dist = node.args[2]
            for row in range(self.n):
                if f == "not_near":
                    if a[row]:
                        out[row] = not_near(self.tokens(row), a[row], b[row], dist)
                elif a[row] and b[row]:
                    out[row] = near(self.tokens(row), a[row], b[row], dist, ordered=(f == "before"))
            return out
        if f == "window":
            inner = self.evaluate(node.args[0])
            k = node.args[1]
            if not self.n:
                return out
            cs = np.r_[0, np.cumsum(inner)]
            idx = np.arange(self.n)
            lo = np.maximum(idx - k, self.doc_first)
            hi = np.minimum(idx + k, self.doc_last)
            return (cs[hi + 1] - cs[lo]) > 0
        cmp = _COMPARE[node.op]
        if f == "doc":
            inner = self.evaluate(node.args[0])
            counts = self._group_count(self.doc, inner)
            return cmp(counts, node.value)
        if f == "doc_share":
            inner = self.evaluate(node.args[0])
            counts = self._group_count(self.doc, inner)
            sizes = self._group_count(self.doc, np.ones(self.n, dtype=bool))
            return cmp(counts / np.maximum(sizes, 1), node.value)
        if f in ("count", "count_distinct"):
            fn = count if f == "count" else count_distinct
            sets = self.set_matches(node.args[0])
            values = np.fromiter((fn(m) if m else 0 for m in sets), dtype=float, count=self.n)
            return cmp(values, node.value)
        raise AssertionError(f)


@dataclass
class AnalysisStats:
    """Counts collected over all batches of an analysis, reported once at the end."""

    excluded: Counter = field(default_factory=Counter)
    emptied_documents: list[str] = field(default_factory=list)
    hierarchy: Counter = field(default_factory=Counter)  # (child, parent) -> sentences
    long_segments: int = 0
    long_examples: list[str] = field(default_factory=list)
    long_limit: int = 0

    def report(self) -> None:
        if self.excluded:
            log.info("Excluded %d sentence(s) by type: %s", sum(self.excluded.values()),
                     ", ".join(f"{k} {v}" for k, v in sorted(self.excluded.items())))
        if self.emptied_documents:
            gone = self.emptied_documents
            log.warning("%d document(s) have no sentences left after the exclusion and are not in the results: %s",
                        len(gone), ", ".join(gone[:10]) + (" ..." if len(gone) > 10 else ""))
        broken = [(c, p, k) for (c, p), k in self.hierarchy.items() if k]
        if broken:
            log.warning(
                "%d child variable(s) are true in sentences where their parent is false (set "
                "aggregate_children = true to make parents include their children): %s",
                len(broken), "; ".join(f"{c} vs parent {p}: {k} sentences" for c, p, k in broken[:5]),
            )
        if self.long_segments:
            log.warning(
                "%d segment(s) are longer than %d words, which usually means sentence splitting "
                "failed (tables, reference lists, text without punctuation). They are slow to "
                "process and distort per-sentence percentages. First: %s",
                self.long_segments, self.long_limit, ", ".join(self.long_examples),
            )


def exclude_sentences(sentences: pd.DataFrame, settings: Settings, stats: AnalysisStats | None = None) -> pd.DataFrame:
    """Drop the sentences whose structure label is in ``exclude_sentence_types``."""
    if not settings.exclude_sentence_types or "structure" not in sentences:
        return sentences
    drop = sentences["structure"].isin(settings.exclude_sentence_types)
    own = stats is None
    stats = stats or AnalysisStats()
    if drop.any():
        stats.excluded.update(sentences.loc[drop, "structure"].value_counts().to_dict())
        stats.emptied_documents += sorted(set(sentences.loc[drop, "document"]) - set(sentences.loc[~drop, "document"]))
    if own:
        stats.report()
    return sentences.loc[~drop].reset_index(drop=True)


def analyze_sentences(
    sentences: pd.DataFrame,
    fw: Framework,
    long_segment_words: int = 500,
    show_progress: bool = False,
    check_hierarchy: bool = True,
    matcher: Matcher | None = None,
    stats: AnalysisStats | None = None,
) -> pd.DataFrame:
    """Code every sentence.

    ``sentences`` needs ``document`` and ``text`` columns, ``vader_compound`` when the
    framework uses sentiment, and optionally ``sentence_id``/``structure``/``start``/``end``,
    which are carried through. The rows of a document must be consecutive and complete:
    DOC, DOC_SHARE and WINDOW look at the whole document. A batch of whole documents can
    therefore be analyzed on its own. Pass ``matcher`` to reuse one between batches, and
    ``stats`` to collect warnings over batches instead of logging them here.
    """
    own_stats = stats is None
    stats = stats if stats is not None else AnalysisStats()
    sentences = sentences.reset_index(drop=True)
    text = [normalize_text(t).lower() for t in sentences["text"].fillna("").astype(str)]
    n = len(text)
    ev = Evaluator(fw, text, sentences["document"].to_numpy())

    matcher = matcher or Matcher(fw.all_lists)
    needed = {name for lists in fw.set_lists.values() for name in lists}
    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        progress = lambda it: tqdm(it, desc="matching", unit=" sentences", leave=False, mininterval=0.5)  # noqa: E731
    hits, kept = matcher.match(text, needed, progress)
    ev.columns.update(hits)
    ev.matches.update(kept)

    compound = None
    if fw.settings.sentiment:
        compound = pd.to_numeric(sentences["vader_compound"], errors="coerce").to_numpy(float)
        if np.isnan(compound).any():
            raise CtrlFindrError("some sentences have no sentiment score; run the ingest step again")
        ev.columns.update(sentiment_flags(compound, fw.settings))

    by_name = {qq.name: qq for qq in fw.queries}
    for name in fw.query_order:
        ev.columns[name] = ev.evaluate(by_name[name].ast)

    by_number = {v.number: v for v in fw.variables}
    for number in fw.variable_order:
        v = by_number[number]
        value = ev.evaluate(v.ast) if v.ast is not None else np.zeros(n, dtype=bool)
        if fw.settings.aggregate_children:
            for child in v.children:
                value = value | ev.variables[child]
        ev.variables[number] = value

    if check_hierarchy and not fw.settings.aggregate_children:
        for v in fw.variables:
            if v.parent:
                stats.hierarchy[(v.number, v.parent)] += int((ev.variables[v.number] & ~ev.variables[v.parent]).sum())

    if long_segment_words and n:
        stats.long_limit = long_segment_words
        lengths = np.fromiter((len(t.split()) for t in text), dtype=int, count=n)
        long_rows = np.flatnonzero(lengths > long_segment_words)
        if len(long_rows):
            ids = sentences["sentence_id"] if "sentence_id" in sentences else pd.Series(np.arange(1, n + 1))
            stats.long_segments += len(long_rows)
            room = 3 - len(stats.long_examples)
            stats.long_examples += [f"{sentences['document'].iat[i]} (sentence {ids.iat[i]}, {lengths[i]} words)"
                                    for i in long_rows[:max(room, 0)]]
    if own_stats:
        stats.report()

    parts: dict[str, object] = {c: sentences[c].to_numpy()
                                for c in ("sentence_id", "structure", "start", "end") if c in sentences}
    parts["document"] = sentences["document"].to_numpy()
    parts["sentences"] = np.array(text, dtype=object)
    for name in fw.lists:
        parts[name] = ev.columns[name]
    if compound is not None:
        parts["vader_compound"] = compound
        for name in SENTIMENT_TERMS:
            parts[name] = ev.columns[name]
    for qq in fw.queries:
        parts[qq.name] = ev.columns[qq.name]
    for v in fw.variables:
        parts[v.column] = ev.variables[v.number]
    return pd.DataFrame(parts)


# ------------------------------------------------------------------------- tests


def run_tests(fw: Framework) -> list[str]:
    """Run the framework's tests sheet; return a message per failed expectation."""
    if not fw.tests:
        return []
    groups = list(dict.fromkeys(t.group for t in fw.tests))
    cases = sorted(fw.tests, key=lambda t: groups.index(t.group))  # rows of a group together
    df = pd.DataFrame({
        "document": [t.group for t in cases],
        "text": [t.sentence for t in cases],
    })
    df["sentence_id"] = df.groupby("document").cumcount() + 1
    if fw.settings.sentiment:
        from .sentiment import score_texts

        df["vader_compound"] = score_texts([normalize_text(t) for t in df["text"]])
    res = analyze_sentences(df, fw, long_segment_words=0, check_hierarchy=False)
    matcher = Matcher(fw.lists)
    columns = {v.number: v.column for v in fw.variables}
    failures = []
    for i, t in enumerate(cases):
        wrong = [(name, True) for name in t.expect_true if not bool(res.at[i, columns.get(name, name)])]
        wrong += [(name, False) for name in t.expect_false if bool(res.at[i, columns.get(name, name)])]
        if not wrong:
            continue
        sentence = res.at[i, "sentences"]
        found = matcher.matches(sentence)
        matched = "; ".join(
            f"{name}: " + ", ".join(sorted({sentence[s:e] for s, e, _ in m}))
            for name, m in zip(matcher.lists, found) if m
        ) or "no list matched"
        for name, expected in wrong:
            detail = _explain(fw, res, i, name)
            failures.append(
                f"tests row {t.row}: expected {name} to be {str(expected).lower()}, got {str(not expected).lower()}. "
                f"Sentence: {t.sentence!r}. Matched: {matched}." + (f" {detail}" if detail else "")
            )
    return failures


def _explain(fw: Framework, res: pd.DataFrame, i: int, name: str) -> str:
    """Values of the names used directly by a failing query or variable."""
    node = None
    var = next((v for v in fw.variables if v.number == name), None)
    if var is not None:
        node = var.ast
    else:
        qq = next((x for x in fw.queries if x.name == name), None)
        node = qq.ast if qq else None
    if node is None:
        return ""
    used = sorted(q.names(node))
    vals = [f"{u}={str(bool(res.at[i, u])).lower()}" for u in used if u in res.columns]
    return f"{name} = {q.to_text(node)}" + (f" where {', '.join(vals)}" if vals else "") + "."
