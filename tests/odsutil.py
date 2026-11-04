"""Write small .ods files for tests (only what ctrlfindr.ods reads)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
)


def _cell(value) -> str:
    if value is None or value == "":
        return "<table:table-cell/>"
    if isinstance(value, tuple):  # (number, displayed text)
        number, shown = value
        return (f'<table:table-cell office:value-type="float" office:value="{number}">'
                f"<text:p>{escape(shown)}</text:p></table:table-cell>")
    return f'<table:table-cell office:value-type="string"><text:p>{escape(str(value))}</text:p></table:table-cell>'


def _row(row: list) -> str:
    """Cells of one row; runs of empty cells use number-columns-repeated, like LibreOffice."""
    out, blanks = [], 0
    for v in row:
        if v is None or v == "":
            blanks += 1
            continue
        if blanks:
            out.append(f'<table:table-cell table:number-columns-repeated="{blanks}"/>')
            blanks = 0
        out.append(_cell(v))
    out.append('<table:table-cell table:number-columns-repeated="1024"/>')  # LibreOffice-style padding
    return "".join(out)


def write_ods(path: Path, sheets: dict[str, list[list]]) -> Path:
    """Runs of empty rows are written as one row with number-rows-repeated, as LibreOffice does."""
    tables = []
    for name, rows in sheets.items():
        parts, blank_rows = [], 0
        for row in rows:
            if not any(v not in (None, "") for v in row):
                blank_rows += 1
                continue
            if blank_rows:
                parts.append(f'<table:table-row table:number-rows-repeated="{blank_rows}"><table:table-cell/></table:table-row>')
                blank_rows = 0
            parts.append("<table:table-row>" + _row(row) + "</table:table-row>")
        parts.append('<table:table-row table:number-rows-repeated="1048000"><table:table-cell/></table:table-row>')
        body = "".join(parts)
        tables.append(f'<table:table table:name="{escape(name)}">{body}</table:table>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {NS} office:version="1.2">'
           f"<office:body><office:spreadsheet>{''.join(tables)}</office:spreadsheet></office:body>"
           "</office:document-content>")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet", compress_type=zipfile.ZIP_STORED)
        zf.writestr("content.xml", xml)
    return path


def framework_sheets(taxonomy: dict[str, list[str]], co=(), dc=(), queries=(), variables=()) -> dict[str, list[list]]:
    """Build the five framework sheets from simple Python structures."""
    names = list(taxonomy)
    depth = max((len(v) for v in taxonomy.values()), default=0)
    tax_rows = [names] + [[taxonomy[n][i] if i < len(taxonomy[n]) else "" for n in names] for i in range(depth)]
    return {
        "How_to_fill": [["instructions"]],
        "variables": [["variable", "variable number", "aggregation query"], *[list(v) for v in variables]],
        "search_strings": [["comment", "query name", "query"], *[["", n, qq] for n, qq in queries]],
        "co_occurrences": [["name of co-occurrence", "first list", "distance between lists", "second list"],
                           *[[n, a, d, b] for n, a, d, b in co]],
        "doc_conditionals": [["name of document-level conditional ", "list"], *[list(x) for x in dc]],
        "taxonomy": tax_rows,
    }
