import json
import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ctrlfindr import analyze, config_from_dict, ingest
from ctrlfindr.common import CtrlFindrError, ValidationError
from ctrlfindr.engine import analyze_sentences
from ctrlfindr.framework import load_framework
from fwutil import make_framework

TAX = [("risk", "risk*"), ("flood", "flood*"), ("ipcc", "ipcc")]
QUERIES = [("q1", "NEAR(risk, flood, 2)"), ("q2", "q1 AND DOC(ipcc)"), ("q3", "flood AND sentiment_negative")]
VARIABLES = [("1", "Risk", "q1"), ("1.1", "IPCC risk", "q2"), ("2", "Bad floods", "q3")]


@pytest.fixture
def small_fw(tmp_path):
    return load_framework(make_framework(tmp_path / "fw.ods", TAX, QUERIES, VARIABLES))


def project(tmp_path, fw_path, text="Flood risk rises. The IPCC agrees.", **cfg):
    (tmp_path / "Text_files").mkdir(exist_ok=True)
    (tmp_path / "Text_files" / "d1.txt").write_text(text, encoding="utf-8")
    return config_from_dict({"paths": {"framework": str(fw_path)}, **cfg}, base=tmp_path)


def test_document_tables(small_fw):
    sents = pd.DataFrame({
        "document": ["d1", "d1", "d2", "d2"],
        "text": ["Flood risk rises.", "The IPCC agrees.", "Risk of terrible flooding.", "Calm day."],
        "vader_compound": [0.2, 0.0, -0.6, 0.3],
    })
    res = analyze_sentences(sents, small_fw)
    docs = analyze.document_outputs(res, small_fw)
    rd = docs["results_document"].set_index("document")
    assert rd.loc["d1"].tolist() == [1, 1, 0] and rd.loc["d2"].tolist() == [1, 0, 1]
    assert docs["results_document_percentage_of_sentences"].set_index("document").loc["d1", "1. Risk"] == 50.0
    sent = docs["results_document_sentiment"].set_index("document")
    assert sent.loc["d2", "2. Bad floods"] == -0.6
    assert np.isnan(sent.loc["d1", "2. Bad floods"])  # never fired in d1 -> empty cell


def test_run_writes_outputs_and_keeps_source_tables(tmp_path, small_fw):
    cfg = project(tmp_path, small_fw.path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)  # a second run still finds its input
    with sqlite3.connect(cfg.paths.database) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"documents", "sentences", "results_sentences"} <= tables
    out = cfg.paths.output
    assert b"\r\n" not in (out / "results_document.tsv").read_bytes()
    assert "## Variables" in (out / "codebook.md").read_text(encoding="utf-8")
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["framework"]["settings"]["sentiment_positive_threshold"] == 0.05
    assert meta["framework"]["meta"]["format_version"] == "1"


def test_run_requires_ingest(tmp_path, small_fw):
    cfg = config_from_dict({"paths": {"framework": str(small_fw.path)}}, base=tmp_path)
    with pytest.raises(CtrlFindrError, match="run the ingest step first"):
        analyze.run(cfg, show_progress=False)


def test_outputs_of_earlier_runs_are_removed(tmp_path, small_fw):
    cfg = project(tmp_path, small_fw.path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    cfg.export.sentence_level = False
    analyze.run(cfg, show_progress=False)
    assert not (cfg.paths.output / "results_sentences.tsv").exists()
    assert (cfg.paths.output / "results_document.tsv").is_file()


def test_failing_framework_tests_stop_or_warn(tmp_path, caplog):
    tests = [("Floods are wonderful, great and lovely.", "q3", "")]  # q3 needs negative sentiment: fails
    path = make_framework(tmp_path / "fw.ods", TAX, QUERIES, VARIABLES, tests=tests)
    cfg = project(tmp_path, path)
    ingest.run(cfg, show_progress=False)
    with pytest.raises(ValidationError, match="framework test"):
        analyze.run(cfg, show_progress=False)
    path = make_framework(tmp_path / "fw2.ods", TAX, QUERIES, VARIABLES, tests=tests, settings={"tests_must_pass": False})
    cfg.paths.framework = path
    with caplog.at_level(logging.WARNING, logger="ctrlfindr"):
        analyze.run(cfg, show_progress=False)
    assert "expected q3 to be true" in caplog.text


def test_long_segments_are_reported(small_fw, caplog):
    sents = pd.DataFrame({"document": ["d"], "text": ["word " * 600], "vader_compound": [0.0]})
    with caplog.at_level(logging.WARNING, logger="ctrlfindr"):
        analyze_sentences(sents, small_fw, long_segment_words=500)
    assert "longer than 500 words" in caplog.text


def test_config_has_no_sentiment_section(tmp_path):
    with pytest.raises(ValidationError, match="framework's settings sheet"):
        config_from_dict({"sentiment": {"enabled": True}}, base=tmp_path)
