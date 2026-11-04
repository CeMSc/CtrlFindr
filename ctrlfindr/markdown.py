"""Split a Markdown document into labelled blocks of plain text.

Each block is ``(structure, text)`` with structure one of:

* ``header``  a heading (``# Title`` or a line underlined with ``===`` / ``---``);
* ``list``    one list item (``-``, ``*``, ``+``, ``1.`` or ``1)``), with its continuation lines;
* ``table``   one table row (``| a | b |``); the separator row is dropped, cells are
              joined with `` | ``;
* ``footnote`` the text of a footnote (``[^1]: ...``), with its indented continuation;
* ``body``    everything else: paragraphs, block quotes, code blocks.

Markdown syntax is removed from the text: emphasis markers, link targets (the link text is
kept), images, inline code backticks, footnote markers (``[^1]``, ``<sup>12</sup>``,
``^12^``; superscript ordinals and exponents are kept as ``20th`` and ``10^6``), HTML tags
and entities, and backslash escapes. Horizontal rules are dropped.
"""

from __future__ import annotations

import html
import re

STRUCTURES = ("header", "body", "list", "table", "footnote")

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$|^ {0,3}(#{1,6})[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_RULE = re.compile(r"^ {0,3}([-*_])([ \t]*\1){2,}[ \t]*$")
_ITEM = re.compile(r"^([ \t]*)([-*+•]|\d{1,9}[.)])[ \t]+(.*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_QUOTE = re.compile(r"^ {0,3}>[ \t]?(.*)$")
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_FOOTNOTE_DEF = re.compile(r"^ {0,3}\[\^[^\]]+\]:[ \t]*(.*)$")
_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*\S+.*$")  # [label]: http://... (link definitions)

_INLINE = [
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),                    # images
    (re.compile(r"!\[[^\]]*\]\[[^\]]*\]"), ""),
    (re.compile(r"\[\^[^\]]+\]"), ""),                             # footnote markers [^1]
    # superscripts (<sup>..</sup> and pandoc ^..^): ordinals are joined ("20th"), a number
    # raised after a digit is an exponent ("10^6"), other number/symbol marks are footnote
    # markers and removed; superscripts with other text keep their text
    (re.compile(r"<sup>\s*(st|nd|rd|th)\s*</sup>|\^(st|nd|rd|th)\^", re.I), r"\1\2"),
    (re.compile(r"(?<=\d)(?:<sup>\s*(\d{1,3})\s*</sup>|\^(\d{1,3})\^)", re.I), r"^\1\2"),
    (re.compile(r"<sup>\s*[\d,;\-–*†‡ ]{1,12}\s*</sup>|\^[\d,\-–*†‡]{1,12}\^", re.I), ""),
    (re.compile(r"<sup>\s*[a-z](\s*,\s*[a-z])*\s*</sup>", re.I), ""),     # letter footnote markers <sup>a</sup>
    (re.compile(r"<sup>(.*?)</sup>", re.I), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),                 # [text](url)
    (re.compile(r"\[([^\]]+)\]\[[^\]]*\]"), r"\1"),                # [text][ref]
    (re.compile(r"<(https?://|www\.)[^>]+>"), ""),                 # <http://autolink>
    (re.compile(r"<!--.*?-->"), ""),                               # HTML comments
    (re.compile(r"</?[A-Za-z][^>]*>"), " "),                       # HTML tags
    (re.compile(r"`+([^`]*)`+"), r"\1"),                           # inline code
    (re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"), r"\2"),         # **bold**, __bold__
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),  # *italic*
    (re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)"), r"\1"),     # _italic_ (not snake_case)
    (re.compile(r"~~(.+?)~~"), r"\1"),                             # ~~strike~~
]
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")  # \* etc.: the character itself, not markup
_PROTECT = 0xE000  # escaped characters are parked in the Unicode private use area meanwhile


def inline_text(text: str) -> str:
    """Plain text of a Markdown line or block (syntax removed, whitespace collapsed)."""
    text = _ESCAPE.sub(lambda m: chr(_PROTECT + ord(m.group(1))), text)
    for rx, repl in _INLINE:
        text = rx.sub(repl, text)
    text = "".join(chr(ord(c) - _PROTECT) if _PROTECT <= ord(c) < _PROTECT + 128 else c for c in text)
    return " ".join(html.unescape(text).split())


def _is_table_row(line: str) -> bool:
    return "|" in line and line.strip() not in ("", "|")


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [inline_text(c) for c in re.split(r"(?<!\\)\|", s)]


def blocks(text: str) -> list[tuple[str, str]]:
    """Parse Markdown into ``(structure, plain text)`` blocks, in document order."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[str, str]] = []
    para: list[str] = []
    item: list[str] | None = None  # current list item or footnote
    item_kind = "list"

    def flush_para() -> None:
        nonlocal para
        if para:
            out.append(("body", " ".join(para)))
            para = []

    def flush_item() -> None:
        nonlocal item
        if item:
            out.append((item_kind, " ".join(item)))
        item = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        fence = _FENCE.match(line)
        if fence:  # code block: kept as body text
            flush_para()
            flush_item()
            marker, code = fence.group(1), []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(marker):
                code.append(lines[i].strip())
                i += 1
            if code:
                out.append(("body", " ".join(c for c in code if c)))
            i += 1
            continue
        if not stripped:
            flush_para()
            flush_item()
            i += 1
            continue
        if not para and item is None and out and out[-1][0] == "footnote" and re.match(r"( {4}|\t)", line):
            item, item_kind = [stripped], "footnote"  # a further paragraph of the same footnote
            i += 1
            continue
        atx = _ATX.match(line)
        if atx:
            flush_para()
            flush_item()
            if atx.group(2):
                out.append(("header", atx.group(2)))
            i += 1
            continue
        if para and item is None and _SETEXT.match(line) and not _ITEM.match(line):
            out.append(("header", " ".join(para)))  # the paragraph above is a setext heading
            para = []
            i += 1
            continue
        if _RULE.match(line):
            flush_para()
            flush_item()
            i += 1
            continue
        if _is_table_row(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]) and "-" in lines[i + 1]:
            flush_para()
            flush_item()
            out.append(("table", " | ".join(c for c in _cells(line) if c)))
            i += 2
            while i < len(lines) and _is_table_row(lines[i]) and lines[i].strip():
                if not _TABLE_SEP.match(lines[i]):
                    row = " | ".join(c for c in _cells(lines[i]) if c)
                    if row:
                        out.append(("table", row))
                i += 1
            continue
        fn = _FOOTNOTE_DEF.match(line)
        if fn:
            flush_para()
            flush_item()
            item, item_kind = [fn.group(1)], "footnote"
            i += 1
            continue
        m = _ITEM.match(line)
        if m and para and item is None and m.group(2)[0].isdigit() and int(m.group(2)[:-1]) != 1:
            m = None  # "2020. Floods..." inside a paragraph is wrapped text, not a list (CommonMark)
        if m:
            flush_para()
            flush_item()
            item, item_kind = [m.group(3)], "list"
            i += 1
            continue
        if item is not None:  # continuation of a list item or footnote (lazy or indented)
            item.append(stripped)
            i += 1
            continue
        q = _QUOTE.match(line)
        if q:
            para.append(q.group(1))
            i += 1
            continue
        if _REF_DEF.match(line):  # [label]: http://... link definition
            i += 1
            continue
        para.append(stripped)
        i += 1
    flush_para()
    flush_item()
    return [(kind, inline_text(t) if kind != "table" else t) for kind, t in out if (inline_text(t) if kind != "table" else t)]
