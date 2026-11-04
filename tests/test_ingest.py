import sqlite3
import sys

import pytest

from ctrlfindr import config_from_dict, ingest
from ctrlfindr.common import CtrlFindrError, ValidationError


def project(tmp_path, files: dict[str, bytes], **ingest_settings):
    text_dir = tmp_path / "Text_files"
    text_dir.mkdir()
    for name, data in files.items():
        (text_dir / name).write_bytes(data)
    return config_from_dict({"ingest": ingest_settings}, base=tmp_path)


def sentences(cfg):
    with sqlite3.connect(cfg.paths.database) as conn:
        return conn.execute('SELECT document, sentence_id, start, "end", text FROM sentences ORDER BY document, sentence_id').fetchall()


def test_clean_document_rules():
    raw = "Heading\nThe security\nclimate is here.\n\n\nNext para  with   spaces… see https://x.org now.[4] The end"
    clean = ingest.clean_document(raw)
    assert clean == "Heading The security climate is here.\n\nNext para with spaces. see now. [4] The end"
    assert "securityclimate" not in ingest.clean_document("Climate security\nClimate change")


def test_config_rejects_fractional_whole_numbers(tmp_path):
    with pytest.raises(ValidationError, match="must be a whole number"):
        config_from_dict({"ingest": {"long_segment_words": 499.9}}, base=tmp_path)
    assert config_from_dict({"ingest": {"long_segment_words": 400.0}}, base=tmp_path).ingest.long_segment_words == 400


def test_crlf_and_bom(tmp_path):
    cfg = project(tmp_path, {"a.txt": "﻿First line.\r\nSecond line.\r\n\r\nNew paragraph.".encode("utf-8")})
    ingest.run(cfg, show_progress=False)
    texts = [r[4] for r in sentences(cfg)]
    assert texts == ["First line.", "Second line.", "New paragraph."]


def test_offsets_point_into_clean_text(tmp_path):
    cfg = project(tmp_path, {"a.txt": b"One sentence here. Another one, Dr. Smith said. [12]\n\nThird."})
    ingest.run(cfg, show_progress=False)
    with sqlite3.connect(cfg.paths.database) as conn:
        clean = conn.execute("SELECT clean_text FROM documents").fetchone()[0]
    rows = sentences(cfg)
    assert [r[4] for r in rows] == ["One sentence here.", "Another one, Dr. Smith said.", "Third."]  # "[12]" dropped
    assert all(clean[s:e] == t for _, _, s, e, t in rows)


def test_discovery_rules(tmp_path):
    cfg = project(tmp_path, {"b.TXT": b"Upper case extension.", "a.report.txt": b"Dotted name.",
                             "c.MD": b"# Title", "notes.docx": b"skip"})
    ingest.run(cfg, show_progress=False)
    assert sorted({r[0] for r in sentences(cfg)}) == ["a.report", "b", "c"]


def test_duplicate_document_names(tmp_path):
    text_dir = tmp_path / "Text_files"
    text_dir.mkdir()
    (text_dir / "a.txt").write_bytes(b"x.")
    (text_dir / "a.Txt").write_bytes(b"y.")
    if len(list(text_dir.iterdir())) < 2:
        pytest.skip("case-insensitive file system")
    with pytest.raises(ValidationError, match="clashing names"):
        ingest.run(config_from_dict({}, base=tmp_path), show_progress=False)


def test_missing_or_empty_folder(tmp_path):
    with pytest.raises(CtrlFindrError, match="not found"):
        ingest.run(config_from_dict({}, base=tmp_path), show_progress=False)
    (tmp_path / "Text_files").mkdir()
    with pytest.raises(CtrlFindrError, match="no .txt or .md files"):
        ingest.run(config_from_dict({}, base=tmp_path), show_progress=False)


def test_non_utf8(tmp_path):
    cfg = project(tmp_path, {"good.txt": b"Fine.", "bad.txt": "Café crème.".encode("cp1252")})
    with pytest.raises(ValidationError, match="not UTF-8") as exc:
        ingest.run(cfg, show_progress=False)
    assert "bad.txt" in str(exc.value)
    cfg.ingest.encoding_errors = "replace"
    ingest.run(cfg, show_progress=False)
    assert any("Caf�" in r[4] for r in sentences(cfg))


def test_empty_file_is_kept_out_of_sentences(tmp_path):
    cfg = project(tmp_path, {"empty.txt": b"", "a.txt": b"Something."})
    ingest.run(cfg, show_progress=False)
    assert {r[0] for r in sentences(cfg)} == {"a"}


def test_incremental(tmp_path):
    cfg = project(tmp_path, {"a.txt": b"Alpha one.", "b.txt": b"Beta one."})
    assert ingest.run(cfg, show_progress=False)["processed"] == 2
    assert ingest.run(cfg, show_progress=False)["processed"] == 0
    (cfg.paths.text_files / "b.txt").write_bytes(b"Beta two. Beta three.")
    (cfg.paths.text_files / "a.txt").unlink()
    (cfg.paths.text_files / "c.txt").write_bytes(b"Gamma.")
    summary = ingest.run(cfg, show_progress=False)
    assert (summary["processed"], summary["removed"], summary["unchanged"]) == (2, 1, 0)
    assert [r[4] for r in sentences(cfg)] == ["Beta two.", "Beta three.", "Gamma."]
    cfg.ingest.text_cleaning = False  # a setting change rebuilds everything
    assert ingest.run(cfg, show_progress=False)["processed"] == 2


def structures(cfg):
    with sqlite3.connect(cfg.paths.database) as conn:
        return conn.execute("SELECT document, structure, text FROM sentences ORDER BY document, sentence_id").fetchall()


def test_txt_is_unstructured_and_md_is_labelled(tmp_path):
    md = ("# Climate *risks*\n\nFloods rose. Droughts too.\n\n- Item one\n- Item two\n\n"
          "| Region | Clashes |\n|---|---|\n| North | 14 |\n\nEnd.").encode()
    cfg = project(tmp_path, {"plain.txt": b"HEADING\nText here. More text.", "brief.md": md})
    ingest.run(cfg, show_progress=False)
    rows = structures(cfg)
    assert [r[1:] for r in rows if r[0] == "brief"] == [
        ("header", "Climate risks"), ("body", "Floods rose."), ("body", "Droughts too."),
        ("list", "Item one"), ("list", "Item two"), ("table", "Region | Clashes"), ("table", "North | 14"),
        ("body", "End.")]
    assert {r[1] for r in rows if r[0] == "plain"} == {"unstructured"}
    with sqlite3.connect(cfg.paths.database) as conn:
        clean = dict(conn.execute("SELECT document, clean_text FROM documents"))
        formats = dict(conn.execute("SELECT document, format FROM documents"))
        spans = conn.execute('SELECT document, start, "end", text FROM sentences').fetchall()
    assert formats == {"brief": "md", "plain": "txt"}
    assert all(clean[d][s:e] == t for d, s, e, t in spans)


def test_txt_and_md_with_the_same_name_clash(tmp_path):
    cfg = project(tmp_path, {"a.txt": b"x.", "a.md": b"y."})
    with pytest.raises(ValidationError, match="clashing names"):
        ingest.run(cfg, show_progress=False)


def test_ingest_does_not_need_vader(tmp_path, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name.startswith("vaderSentiment") else real(name, *a))
    monkeypatch.setitem(sys.modules, "vaderSentiment", None)
    cfg = project(tmp_path, {"a.txt": b"Fine text."})
    assert ingest.run(cfg, show_progress=False)["sentences"] == 1
