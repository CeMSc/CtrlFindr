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


def write_ods(path: Path, sheets: dict[str, list[list]]) -> Path:
    tables = []
    for name, rows in sheets.items():
        body = "".join("<table:table-row>" + "".join(_cell(v) for v in row) + "</table:table-row>" for row in rows)
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
