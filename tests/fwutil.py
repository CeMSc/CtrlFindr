"""Build small format-1 frameworks for tests."""

from __future__ import annotations

from pathlib import Path

from ctrlfindr.spreadsheet import SheetSpec, write_workbook


def make_framework(
    path: Path,
    taxonomy: list[tuple] = (),
    queries: list[tuple] = (),
    variables: list[tuple] = (),
    settings: dict | None = None,
    tests: list[tuple] = (),
    meta: dict | None = None,
    drop_sheets: tuple[str, ...] = (),
) -> Path:
    """taxonomy: (list, term[, note]); queries: (name, query[, note]);
    variables: (number, name, definition[, parent]); tests: (sentence, expect_true, expect_false[, group, paragraph])."""
    meta = {"format_version": "1", **(meta or {})}
    sheets = {
        "meta": [["key", "value"]] + [[k, v] for k, v in meta.items()],
        "settings": [["key", "value", "note"]] + [[k, str(v).lower() if isinstance(v, bool) else str(v), ""]
                                                   for k, v in (settings or {}).items()],
        "taxonomy": [["list", "term", "note"]] + [list(r) + [""] * (3 - len(r)) for r in taxonomy],
        "queries": [["name", "query", "note"]] + [list(r) + [""] * (3 - len(r)) for r in queries],
        "variables": [["number", "name", "definition", "parent", "note"]] + [list(r) + [""] * (5 - len(r)) for r in variables],
        "tests": [["sentence", "expect_true", "expect_false", "group", "paragraph", "note"]]
                 + [list(r) + [""] * (6 - len(r)) for r in tests],
    }
    specs = [SheetSpec(name, rows) for name, rows in sheets.items() if name not in drop_sheets]
    return write_workbook(path, specs)
