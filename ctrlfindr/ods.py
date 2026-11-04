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

# Spreadsheets pad sheets with huge "repeated" blank rows/columns at the end. Blank runs
# are therefore expanded only when content follows them, so that row and column numbers
# stay exact while trailing padding costs nothing.


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
        blank_rows = 0
        for row in table.iter(f"{_T}table-row"):
            cells: list[str] = []
            blank_cells = 0
            for cell in row:
                if cell.tag not in (f"{_T}table-cell", f"{_T}covered-table-cell"):
                    continue
                text = _cell_text(cell)
                repeat = int(cell.get(f"{_T}number-columns-repeated", "1"))
                if not text.strip():
                    blank_cells += repeat
                    continue
                cells.extend([""] * blank_cells)
                blank_cells = 0
                cells.extend([text] * repeat)
            repeat_rows = int(row.get(f"{_T}number-rows-repeated", "1"))
            if not cells:
                blank_rows += repeat_rows
                continue
            rows.extend([] for _ in range(blank_rows))
            blank_rows = 0
            rows.extend(list(cells) for _ in range(repeat_rows))
        sheets[table.get(f"{_T}name", "")] = rows
    return sheets


# ------------------------------------------------------------------------ writing

_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:of="urn:oasis:names:tc:opendocument:xmlns:of:1.2"'
)
_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">'
    '<manifest:file-entry manifest:full-path="/" manifest:version="1.2" '
    'manifest:media-type="application/vnd.oasis.opendocument.spreadsheet"/>'
    '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
    '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>'
    "</manifest:manifest>"
)
_STYLES = f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {_NS} office:version="1.2"/>'


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _para(line: str) -> str:
    # Runs of spaces must be written as <text:s/> to survive.
    out = []
    for i, part in enumerate(line.split("  ")):
        if i:
            out.append(' <text:s/>')
        out.append(_esc(part))
    return f"<text:p>{''.join(out)}</text:p>"


def write_ods(path: Path, sheets) -> None:
    """Write ``SheetSpec`` objects (see ctrlfindr.spreadsheet) to an .ods file."""
    styles = [
        '<style:style style:name="ceH" style:family="table-cell"><style:text-properties fo:font-weight="bold"/></style:style>',
        '<style:style style:name="ceW" style:family="table-cell"><style:table-cell-properties '
        'fo:wrap-option="wrap" style:vertical-align="top"/></style:style>',
        '<style:style style:name="ceHW" style:family="table-cell"><style:table-cell-properties '
        'fo:wrap-option="wrap" style:vertical-align="top"/><style:text-properties fo:font-weight="bold"/></style:style>',
    ]
    widths: dict[float, str] = {}
    validations: list[str] = []
    tables: list[str] = []
    for spec in sheets:
        cols = []
        for w in spec.widths_cm:
            if w not in widths:
                widths[w] = f"co{len(widths) + 1}"
                styles.append(f'<style:style style:name="{widths[w]}" style:family="table-column">'
                              f'<style:table-column-properties style:column-width="{w}cm"/></style:style>')
            cols.append(f'<table:table-column table:style-name="{widths[w]}"/>')
        dd_names: dict[int, tuple[str, int]] = {}
        for dd in spec.dropdowns:
            name = f"val{len(validations) + 1}"
            if dd.source:
                sheet, col = dd.source
                cond = f"of:cell-content-is-in-list([$'{sheet}'.${col}$2:.${col}${dd.rows + 1}])"
            else:
                cond = "of:cell-content-is-in-list(" + ";".join(f'"{v}"' for v in dd.values) + ")"
            validations.append(
                f'<table:content-validation table:name="{name}" table:condition="{_esc(cond)}" '
                f'table:allow-empty-cell="true" table:display-list="unsorted" '
                f"table:base-cell-address=\"$'{_esc(spec.name)}'.$A$2\">"
                '<table:error-message table:display="false"/></table:content-validation>'
            )
            dd_names[dd.column] = (name, dd.rows)
        width = max([len(r) for r in spec.rows] + [max(dd_names, default=-1) + 1, len(spec.widths_cm), 1])
        rows_xml = []

        def cell(value: str, r: int, c: int) -> str:
            attrs = ""
            if c in dd_names and 1 <= r <= dd_names[c][1]:
                attrs += f' table:content-validation-name="{dd_names[c][0]}"'
            style = ("ceHW" if spec.wrap else "ceH") if (spec.header and r == 0) else ("ceW" if spec.wrap else "")
            if style:
                attrs += f' table:style-name="{style}"'
            if value == "":
                return f"<table:table-cell{attrs}/>"
            body = "".join(_para(line) for line in str(value).split("\n"))
            return f'<table:table-cell office:value-type="string"{attrs}>{body}</table:table-cell>'

        for r, row in enumerate(spec.rows):
            cells = [cell(row[c] if c < len(row) else "", r, c) for c in range(width)]
            rows_xml.append("<table:table-row>" + "".join(cells) + "</table:table-row>")
        last_dd_row = max((rows for _, rows in dd_names.values()), default=0)
        remaining = last_dd_row + 1 - len(spec.rows)
        if remaining > 0:  # empty rows that still carry the dropdowns
            cells = [cell("", len(spec.rows), c) for c in range(width)]
            rows_xml.append(f'<table:table-row table:number-rows-repeated="{remaining}">' + "".join(cells) + "</table:table-row>")
        tables.append(f'<table:table table:name="{_esc(spec.name)}">' + "".join(cols) + "".join(rows_xml) + "</table:table>")

    content = (
        f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {_NS} office:version="1.2">'
        f"<office:automatic-styles>{''.join(styles)}</office:automatic-styles>"
        "<office:body><office:spreadsheet>"
        + (f"<table:content-validations>{''.join(validations)}</table:content-validations>" if validations else "")
        + "".join(tables)
        + "</office:spreadsheet></office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.oasis.opendocument.spreadsheet",
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", _MANIFEST)
        zf.writestr("content.xml", content)
        zf.writestr("styles.xml", _STYLES)
