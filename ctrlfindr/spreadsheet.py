"""Read and write framework workbooks in OpenDocument (.ods) or Excel (.xlsx) format.

Reading returns every cell as text, the way it is displayed, so that identifiers such as
variable number ``1.10`` are not turned into numbers. Writing is used for the templates
and the tests; it supports bold headers, column widths, text-formatted columns and
dropdown lists (data validation).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from . import ods

SUFFIXES = (".ods", ".xlsx")


@dataclass
class Dropdown:
    """A dropdown on one column: fixed ``values`` or the cells of ``source`` (sheet, column letter)."""

    column: int  # 0-based
    values: tuple[str, ...] = ()
    source: tuple[str, str] | None = None
    rows: int = 500  # how many rows below the header get the dropdown


@dataclass
class SheetSpec:
    name: str
    rows: list[list[str]]
    widths_cm: list[float] = field(default_factory=list)
    header: bool = True  # first row bold
    wrap: bool = False  # wrap text in all cells
    dropdowns: list[Dropdown] = field(default_factory=list)


def read_workbook(path: str | Path) -> dict[str, list[list[str]]]:
    """``{sheet name: rows}`` with cells as strings; trailing empty cells/rows removed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ods":
        return ods.read_ods(path)
    if suffix == ".xlsx":
        return _read_xlsx(path)
    raise ValueError(f"unsupported file type {path.suffix!r}; use .ods or .xlsx")


def write_workbook(path: str | Path, sheets: list[SheetSpec]) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ods":
        ods.write_ods(path, sheets)
    elif suffix == ".xlsx":
        _write_xlsx(path, sheets)
    else:
        raise ValueError(f"unsupported file type {path.suffix!r}; use .ods or .xlsx")
    return path


# ------------------------------------------------------------------------- xlsx

def _xlsx_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    return str(value)


def _read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets: dict[str, list[list[str]]] = {}
        for ws in wb.worksheets:
            rows: list[list[str]] = []
            pending_blank = 0
            for values in ws.iter_rows(values_only=True):
                cells = [_xlsx_text(v) for v in values]
                while cells and not cells[-1].strip():
                    cells.pop()
                if not cells:
                    pending_blank += 1
                    continue
                rows.extend([] for _ in range(pending_blank))
                pending_blank = 0
                rows.append(cells)
            sheets[ws.title] = rows
        return sheets
    finally:
        wb.close()


def _col_letter(index: int) -> str:
    from openpyxl.utils import get_column_letter

    return get_column_letter(index + 1)


def _write_xlsx(path: Path, sheets: list[SheetSpec]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    wb.remove(wb.active)
    for spec in sheets:
        ws = wb.create_sheet(spec.name)
        width = max([len(r) for r in spec.rows] + [len(spec.widths_cm), 1])
        for r, row in enumerate(spec.rows, start=1):
            for c, value in enumerate(row, start=1):
                cell = ws.cell(row=r, column=c, value=value if value != "" else None)
                cell.number_format = "@"  # text: Excel must not turn 1.10 into 1.1
                if spec.wrap:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                if spec.header and r == 1:
                    cell.font = Font(bold=True)
        # Keep typed values as text in empty rows too.
        for c in range(1, width + 1):
            for r in range(len(spec.rows) + 1, len(spec.rows) + 501):
                ws.cell(row=r, column=c).number_format = "@"
        for c, w in enumerate(spec.widths_cm):
            ws.column_dimensions[_col_letter(c)].width = w * 5  # ~characters per cm
        if spec.header:
            ws.freeze_panes = "A2"
        for dd in spec.dropdowns:
            if dd.source:
                sheet, col = dd.source
                formula = f"'{sheet}'!${col}$2:${col}${dd.rows + 1}"
            else:
                formula = '"' + ",".join(dd.values) + '"'
            dv = DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=False)
            letter = _col_letter(dd.column)
            dv.add(f"{letter}2:{letter}{dd.rows + 1}")
            ws.add_data_validation(dv)
    wb.save(path)
