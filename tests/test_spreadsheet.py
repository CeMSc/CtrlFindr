import zipfile

import pytest

from ctrlfindr.framework import load_framework
from ctrlfindr.common import ValidationError
from ctrlfindr.spreadsheet import Dropdown, SheetSpec, read_workbook, write_workbook
from ctrlfindr.template import write_template

ROWS = [["key", "value"], ["number", "1.10"], ["spaces", "a  b"], ["lines", "one\ntwo"], [], ["after gap", "x"]]


@pytest.mark.parametrize("ext", ["ods", "xlsx"])
def test_round_trip_keeps_text_and_row_numbers(tmp_path, ext):
    path = write_workbook(tmp_path / f"t.{ext}", [SheetSpec("s", ROWS, dropdowns=[Dropdown(1, ("true", "false"))])])
    rows = read_workbook(path)["s"]
    assert rows[:4] == ROWS[:4]
    assert rows[4] == [] and rows[5] == ["after gap", "x"]  # row 6 stays row 6


def test_xlsx_numbers_and_booleans(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "s"
    ws.append(["a", 1, 2.5, True, None, 3.0])
    wb.save(tmp_path / "n.xlsx")
    assert read_workbook(tmp_path / "n.xlsx")["s"] == [["a", "1", "2.5", "true", "", "3"]]


def test_ods_displayed_text_is_read(tmp_path):
    # A numeric cell typed as 1.10 is stored as 1.1 but displayed as "1.10".
    xml = ('<?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
           'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
           '<office:body><office:spreadsheet><table:table table:name="s"><table:table-row>'
           '<table:table-cell office:value-type="float" office:value="1.1"><text:p>1.10</text:p></table:table-cell>'
           '</table:table-row></table:table></office:spreadsheet></office:body></office:document-content>')
    with zipfile.ZipFile(tmp_path / "d.ods", "w") as zf:
        zf.writestr("content.xml", xml)
    assert read_workbook(tmp_path / "d.ods")["s"] == [["1.10"]]


@pytest.mark.parametrize("ext", ["ods", "xlsx"])
def test_blank_template_only_lacks_content(tmp_path, ext):
    path = write_template(tmp_path / f"t.{ext}")
    with pytest.raises(ValidationError) as exc:
        load_framework(path)
    assert exc.value.problems == ["sheet 'taxonomy': no lists defined"]


def test_unsupported_file_type(tmp_path):
    (tmp_path / "f.csv").write_text("x")
    with pytest.raises(ValidationError, match="unsupported file type"):
        load_framework(tmp_path / "f.csv")
