"""Minimal reader for OpenDocument spreadsheets (.ods), standard library only.

Cells are returned as the text shown in the spreadsheet, never as parsed numbers.
This matters for the framework: a variable number typed as ``1.10`` stays ``"1.10"``
instead of turning into the float ``1.1`` (which would collide with variable ``1.1``).
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_T = f"{{{_TABLE}}}"
_X = f"{{{_TEXT}}}"

# Spreadsheets pad sheets with huge "repeated" blank rows/columns; never expand those.
_MAX_REPEAT_BLANK = 1


def _paragraph_text(p: ET.Element) -> str:
    """Text of a <text:p>, honouring <text:s/> (spaces), tabs and line breaks."""
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        if el.text:
            parts.append(el.text)
        for child in el:
            if child.tag == f"{_X}s":
                parts.append(" " * int(child.get(f"{_X}c", "1")))
            elif child.tag == f"{_X}tab":
                parts.append("\t")
            elif child.tag == f"{_X}line-break":
                parts.append("\n")
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(p)
    return "".join(parts)


def _cell_text(cell: ET.Element) -> str:
    return "\n".join(_paragraph_text(p) for p in cell.findall(f"{_X}p"))


def read_ods(path: str | Path) -> dict[str, list[list[str]]]:
    """Return ``{sheet name: rows}``; each row is a list of cell strings ('' for empty).

    Trailing empty cells and trailing empty rows are removed.
    """
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("content.xml"))
    sheets: dict[str, list[list[str]]] = {}
    for table in root.iter(f"{_T}table"):
        rows: list[list[str]] = []
        for row in table.iter(f"{_T}table-row"):
            cells: list[str] = []
            for cell in row:
                if cell.tag not in (f"{_T}table-cell", f"{_T}covered-table-cell"):
                    continue
                text = _cell_text(cell)
                repeat = int(cell.get(f"{_T}number-columns-repeated", "1"))
                if not text:
                    repeat = min(repeat, _MAX_REPEAT_BLANK) if repeat > 1000 else repeat
                cells.extend([text] * repeat)
            while cells and not cells[-1].strip():
                cells.pop()
            repeat_rows = int(row.get(f"{_T}number-rows-repeated", "1"))
            if not cells:
                repeat_rows = min(repeat_rows, _MAX_REPEAT_BLANK)
            rows.extend([list(cells) for _ in range(repeat_rows)])
        while rows and not rows[-1]:
            rows.pop()
        sheets[table.get(f"{_T}name", "")] = rows
    return sheets
