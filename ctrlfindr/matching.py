"""Keyword matching (one Aho-Corasick pass over all taxonomy terms) and co-occurrences.

Matching rules (documented in the README):

* matching is case-insensitive (sentences and terms are lowercased);
* ``*`` stands for any run of letters, digits or underscores, including none
  (``increas*`` matches ``increase``, ``increasing``; it does not cross a hyphen or space);
* a term only matches whole words: the character before the match and the character
  after it must not be a letter, digit or underscore. This is the same as wrapping the
  term in ``(?<!\\w)`` and ``(?!\\w)`` in a Python regular expression.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass

import ahocorasick
import numpy as np

Span = tuple[int, int]
_WORDCHARS = re.compile(r"\w*")
_WORD_TOKEN = re.compile(r"\S*\w\S*")  # whitespace-separated token with a letter or digit


def _isw(ch: str) -> bool:
    """Python's definition of ``\\w`` for str patterns."""
    return ch.isalnum() or ch == "_"


def term_regex(term: str) -> re.Pattern[str]:
    """Regular expression with the same semantics as the automaton (used for odd wildcards)."""
    body = re.escape(term).replace(r"\*", r"\w*")
    return re.compile(rf"(?<!\w){body}(?!\w)")


@dataclass(frozen=True)
class _Entry:
    list_idx: int
    kind: str  # "exact" | "prefix" (only trailing *) | "regex" (other wildcards)
    regex: re.Pattern[str] | None


class Matcher:
    """Find every taxonomy list in every sentence with a single scan per sentence."""

    def __init__(self, taxonomy: dict[str, list[str]]):
        self.lists = list(taxonomy)
        self._automaton = ahocorasick.Automaton()
        entries: dict[str, list[_Entry]] = defaultdict(list)
        self._scan: list[tuple[int, re.Pattern[str]]] = []  # terms with a leading wildcard
        for idx, name in enumerate(self.lists):
            for term in taxonomy[name]:
                star = term.find("*")
                if star == -1:
                    entries[term].append(_Entry(idx, "exact", None))
                elif star == len(term) - 1 and term.count("*") == 1:
                    entries[term[:-1]].append(_Entry(idx, "prefix", None))
                elif star == 0:
                    self._scan.append((idx, term_regex(term)))
                else:
                    entries[term[:star]].append(_Entry(idx, "regex", term_regex(term)))
        for literal, items in entries.items():
            self._automaton.add_word(literal, (literal, tuple(items)))
        if entries:
            self._automaton.make_automaton()
        self._empty = not entries

    def spans(self, sentence: str) -> list[set[Span]]:
        """For one (lowercased) sentence, the set of match spans of each list."""
        found: list[set[Span]] = [set() for _ in self.lists]
        n = len(sentence)
        if not self._empty:
            for end, (literal, items) in self._automaton.iter(sentence):
                start = end - len(literal) + 1
                if start > 0 and _isw(sentence[start - 1]):
                    continue
                after = end + 1
                for e in items:
                    if e.kind == "exact":
                        if after < n and _isw(sentence[after]):
                            continue
                        found[e.list_idx].add((start, after))
                    elif e.kind == "prefix":
                        found[e.list_idx].add((start, _WORDCHARS.match(sentence, after).end()))
                    else:
                        m = e.regex.match(sentence, start)
                        if m:
                            found[e.list_idx].add(m.span())
        for idx, rx in self._scan:
            for m in rx.finditer(sentence):
                found[idx].add(m.span())
        return found

    def match(
        self, sentences: list[str], keep_spans: set[str] | None = None, progress=None
    ) -> tuple[dict[str, np.ndarray], dict[str, list[list[Span] | None]]]:
        """Boolean column per list, plus sorted spans for the lists in ``keep_spans``."""
        keep = [i for i, name in enumerate(self.lists) if keep_spans and name in keep_spans]
        hits = np.zeros((len(self.lists), len(sentences)), dtype=bool)
        spans: dict[str, list[list[Span] | None]] = {self.lists[i]: [None] * len(sentences) for i in keep}
        iterator = sentences if progress is None else progress(sentences)
        for row, sentence in enumerate(iterator):
            found = self.spans(sentence)
            for idx, s in enumerate(found):
                if s:
                    hits[idx, row] = True
            for idx in keep:
                if found[idx]:
                    spans[self.lists[idx]][row] = sorted(found[idx])
        return {name: hits[i] for i, name in enumerate(self.lists)}, spans


def words_between(starts: list[int], ends: list[int], left_end: int, right_start: int) -> int:
    """Number of word tokens lying entirely between two character positions."""
    first = bisect.bisect_left(starts, left_end)
    last = bisect.bisect_right(ends, right_start)
    return max(0, last - first)


def co_occurs(sentence: str, spans1: list[Span], spans2: list[Span], distance: int) -> bool:
    """True if a match of list 1 and a match of list 2 are at most ``distance`` words apart.

    The distance is the number of words strictly between the end of the earlier match and
    the start of the later one (adjacent matches have distance 0). Order does not matter.
    Overlapping matches (for example the same word matched by both lists) never count.
    Words are whitespace-separated tokens that contain a letter or digit, so standalone
    punctuation such as "-" is not counted.
    """
    tokens = [m.span() for m in _WORD_TOKEN.finditer(sentence)]
    starts = [t[0] for t in tokens]
    ends = [t[1] for t in tokens]
    for left, right in ((spans1, spans2), (spans2, spans1)):
        right_starts = [s[0] for s in right]
        for _, l_end in left:
            j = bisect.bisect_left(right_starts, l_end)  # nearest match starting after `left`
            if j < len(right) and words_between(starts, ends, l_end, right_starts[j]) <= distance:
                return True
    return False


def co_occurrence_column(
    sentences: list[str],
    spans1: list[list[Span] | None],
    spans2: list[list[Span] | None],
    distance: int,
) -> np.ndarray:
    """Evaluate one co-occurrence rule; only sentences where both lists matched are checked."""
    out = np.zeros(len(sentences), dtype=bool)
    for i, (a, b) in enumerate(zip(spans1, spans2)):
        if a and b:
            out[i] = co_occurs(sentences[i], a, b, distance)
    return out
