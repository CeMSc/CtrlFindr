"""Term syntax, keyword matching (one Aho-Corasick pass for all terms), proximity and counts.

Term syntax (terms are lowercased and normalized like the corpus):

* ``flood``           the word "flood" only (whole words: the characters around a match
                      must not be letters, digits or underscores);
* ``flood*``          ``*`` = any run of letters/digits/underscores, also empty, anywhere in a
                      term (``*fire``, ``biodivers* loss``); it never crosses a space or hyphen;
* ``human~wildlife``  ``~`` = a hyphen, a space or nothing ("human-wildlife",
                      "human wildlife", "humanwildlife");
* ``climate _ risk``  a ``_`` standing alone = exactly one word ("climate change risk");
* ``-business climate``   an exclusion phrase: a match of the list that lies inside this
                      phrase is ignored (other matches in the same sentence still count);
* ``-@other_list``    every term of ``other_list`` is an exclusion phrase for this list.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass, field

import ahocorasick
import numpy as np

Span = tuple[int, int]
Match = tuple[int, int, int]  # start, end, entry id
Tokens = tuple[list[int], list[int]]  # start and end offsets of the words of a sentence

_WORDCHARS = re.compile(r"\w*")
_TOKEN = re.compile(r"\S+")
_HAS_WORDCHAR = re.compile(r"\w")


class TermError(ValueError):
    pass


def _isw(ch: str) -> bool:
    """Python's definition of ``\\w`` for str patterns."""
    return ch.isalnum() or ch == "_"


# ------------------------------------------------------------------- term syntax


@dataclass(frozen=True)
class Term:
    text: str
    regex: re.Pattern[str]
    literal: str  # literal text before the first marker (used by the automaton)
    kind: str  # "exact" | "prefix" (single trailing *) | "regex" | "scan" (no literal prefix)


def parse_term(text: str) -> Term:
    """Compile one (already lowercased and normalized) term; raises TermError."""
    words = text.split(" ")
    if not _HAS_WORDCHAR.search(text.replace("*", "").replace("~", "").replace(" _ ", " ")) or text == "_":
        raise TermError(f"term {text!r} has no letters or digits")
    if words[0] == "_" or words[-1] == "_":
        raise TermError(f"term {text!r}: a word gap '_' needs words on both sides")
    if text.startswith("~") or text.endswith("~") or "~~" in text or "~ " in text or " ~" in text:
        raise TermError(f"term {text!r}: '~' must join two parts of a word, as in human~wildlife")
    parts = []
    for i, w in enumerate(words):
        if i:
            parts.append(" ")
        if w == "_":
            parts.append(r"\S+")
            continue
        for ch in w:
            parts.append(r"\w*" if ch == "*" else "[- ]?" if ch == "~" else re.escape(ch))
    regex = re.compile(r"(?<!\w)" + "".join(parts) + r"(?!\w)")

    special = [i for i in (text.find("*"), text.find("~"), (" " + text + " ").find(" _ ")) if i >= 0]
    cut = min(special) if special else len(text)
    literal = text[:cut]
    if not special:
        kind = "exact"
    elif text.count("*") == 1 and text.endswith("*") and "~" not in text and " _ " not in f" {text} ":
        kind = "prefix"
    elif literal:
        kind = "regex"
    else:
        kind = "scan"
    return Term(text, regex, literal, kind)


def term_regex(text: str) -> re.Pattern[str]:
    return parse_term(text).regex


@dataclass
class ListSpec:
    """A list of the taxonomy: positive terms plus optional exclusions."""

    terms: list[str]
    exclude_terms: list[str] = field(default_factory=list)
    exclude_lists: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Entry:
    slot: int  # internal list slot
    entry: int  # entry id, unique across all lists (for COUNT_DISTINCT over several lists)
    kind: str
    regex: re.Pattern[str] | None


class Matcher:
    """Find every list (with exclusions applied) in a sentence with one automaton scan."""

    def __init__(self, lists: dict[str, ListSpec]):
        self.lists = list(lists)
        # slots: one per list, plus one per list with inline exclusion phrases
        slots: list[str] = list(self.lists)
        self._excl_slot: dict[int, int] = {}
        for i, name in enumerate(self.lists):
            if lists[name].exclude_terms:
                self._excl_slot[i] = len(slots)
                slots.append(f"{name} (exclusions)")
        self._nslots = len(slots)
        index = {name: i for i, name in enumerate(self.lists)}
        self._excl_lists = {i: [index[x] for x in lists[name].exclude_lists] for i, name in enumerate(self.lists)}
        self._order = self._exclusion_order()

        self._automaton = ahocorasick.Automaton()
        entries: dict[str, list[_Entry]] = defaultdict(list)
        self._scan: list[tuple[int, int, re.Pattern[str]]] = []

        def add(slot: int, entry: int, text: str) -> None:
            t = parse_term(text)
            if t.kind == "scan":
                self._scan.append((slot, entry, t.regex))
            else:
                entries[t.literal].append(_Entry(slot, entry, t.kind, t.regex if t.kind == "regex" else None))

        next_id = 0
        for i, name in enumerate(self.lists):
            for text in lists[name].terms:
                add(i, next_id, text)
                next_id += 1
            for text in lists[name].exclude_terms:
                add(self._excl_slot[i], -1, text)
        for literal, items in entries.items():
            self._automaton.add_word(literal, (literal, tuple(items)))
        self._empty = not entries
        if entries:
            self._automaton.make_automaton()

    def _exclusion_order(self) -> list[int]:
        """Lists ordered so that exclusion lists are finished before the lists using them."""
        order: list[int] = []
        state: dict[int, int] = {}

        def visit(i: int, path: list[int]) -> None:
            if state.get(i) == 2:
                return
            if state.get(i) == 1:
                cyc = " -> ".join(self.lists[k] for k in path[path.index(i):] + [i])
                raise TermError(f"lists exclude each other in a loop: {cyc}")
            state[i] = 1
            for j in self._excl_lists[i]:
                visit(j, path + [i])
            state[i] = 2
            order.append(i)

        for i in range(len(self.lists)):
            visit(i, [])
        return order

    def matches(self, sentence: str) -> list[list[Match]]:
        """For one (lowercased) sentence: the matches of each list after exclusions."""
        raw: list[set[Match]] = [set() for _ in range(self._nslots)]
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
                        raw[e.slot].add((start, after, e.entry))
                    elif e.kind == "prefix":
                        raw[e.slot].add((start, _WORDCHARS.match(sentence, after).end(), e.entry))
                    else:
                        m = e.regex.match(sentence, start)
                        if m:
                            raw[e.slot].add((m.start(), m.end(), e.entry))
        for slot, entry, rx in self._scan:
            for m in rx.finditer(sentence):
                raw[slot].add((m.start(), m.end(), entry))

        final: list[list[Match]] = [[] for _ in self.lists]
        for i in self._order:
            found = raw[i]
            excl: list[Span] = []
            if i in self._excl_slot:
                excl += [(s, e) for s, e, _ in raw[self._excl_slot[i]]]
            for j in self._excl_lists[i]:
                excl += [(s, e) for s, e, _ in final[j]]
            if excl and found:
                found = {m for m in found if not any(s <= m[0] and m[1] <= e for s, e in excl)}
            final[i] = sorted(found)
        return final

    def match(
        self, sentences: list[str], keep: set[str] | None = None, progress=None
    ) -> tuple[dict[str, np.ndarray], dict[str, list[list[Match] | None]]]:
        """Boolean column per list, plus the matches of the lists in ``keep``."""
        keep_idx = [i for i, name in enumerate(self.lists) if keep and name in keep]
        hits = np.zeros((len(self.lists), len(sentences)), dtype=bool)
        kept: dict[str, list[list[Match] | None]] = {self.lists[i]: [None] * len(sentences) for i in keep_idx}
        iterator = sentences if progress is None else progress(sentences)
        for row, sentence in enumerate(iterator):
            found = self.matches(sentence)
            for idx, m in enumerate(found):
                if m:
                    hits[idx, row] = True
            for idx in keep_idx:
                if found[idx]:
                    kept[self.lists[idx]][row] = found[idx]
        return {name: hits[i] for i, name in enumerate(self.lists)}, kept


# -------------------------------------------------------------- words and proximity


def word_tokens(sentence: str) -> Tokens:
    """Offsets of the whitespace-separated tokens that contain a letter or digit."""
    starts: list[int] = []
    ends: list[int] = []
    for m in _TOKEN.finditer(sentence):
        if _HAS_WORDCHAR.search(m.group()):
            starts.append(m.start())
            ends.append(m.end())
    return starts, ends


def words_between(tokens: Tokens, left_end: int, right_start: int) -> int:
    """Number of words lying entirely between two character positions."""
    starts, ends = tokens
    return max(0, bisect.bisect_right(ends, right_start) - bisect.bisect_left(starts, left_end))


def _nearest_after(tokens: Tokens, end: int, others: list[Match], other_starts: list[int]) -> int | None:
    """Words to the nearest match in ``others`` that starts at or after ``end``."""
    j = bisect.bisect_left(other_starts, end)
    return words_between(tokens, end, other_starts[j]) if j < len(others) else None


def _nearest_before(tokens: Tokens, start: int, other_ends: list[int]) -> int | None:
    """Words to the nearest match that ends at or before ``start`` (``other_ends`` sorted)."""
    j = bisect.bisect_right(other_ends, start)
    return words_between(tokens, other_ends[j - 1], start) if j else None


def near(tokens: Tokens, a: list[Match], b: list[Match], distance: int, ordered: bool = False) -> bool:
    """A match of ``a`` and a match of ``b`` at most ``distance`` words apart.

    Distance = words strictly between the end of the earlier and the start of the later
    match (adjacent = 0). Overlapping matches never count. With ``ordered``, ``a`` must
    come first.
    """
    b_starts = [m[0] for m in b]
    for _, end, _ in a:
        d = _nearest_after(tokens, end, b, b_starts)
        if d is not None and d <= distance:
            return True
    if ordered:
        return False
    a_starts = [m[0] for m in a]
    for _, end, _ in b:
        d = _nearest_after(tokens, end, a, a_starts)
        if d is not None and d <= distance:
            return True
    return False


def not_near(tokens: Tokens, a: list[Match], b: list[Match], distance: int) -> bool:
    """Some match of ``a`` has no match of ``b`` within ``distance`` words (on either side)."""
    if not a:
        return False
    if not b:
        return True
    b_starts = [m[0] for m in b]
    b_ends = sorted(m[1] for m in b)
    for start, end, _ in a:
        after = _nearest_after(tokens, end, b, b_starts)
        before = _nearest_before(tokens, start, b_ends)
        if not any(d is not None and d <= distance for d in (after, before)):
            return True
    return False


def longest_matches(matches: list[Match]) -> list[Match]:
    """Drop matches lying inside a longer match (and duplicates of the same span)."""
    out: list[Match] = []
    # longest first, then by position, so a containing match is seen before its parts
    for m in sorted(matches, key=lambda x: (x[0] - x[1], x[0])):
        if not any(o[0] <= m[0] and m[1] <= o[1] for o in out):
            out.append(m)
    return sorted(out)


def count(matches: list[Match]) -> int:
    return len(longest_matches(matches))


def count_distinct(matches: list[Match]) -> int:
    return len({m[2] for m in longest_matches(matches)})
