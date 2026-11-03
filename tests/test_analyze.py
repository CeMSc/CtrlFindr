import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from ctrlfindr import analyze, config_from_dict, ingest
from ctrlfindr.common import CtrlFindrError
from ctrlfindr.config import SentimentSettings
from ctrlfindr.framework import load_framework
from odsutil import framework_sheets, write_ods


def test_sentiment_flags_default_and_v21_thresholds():
    c = np.array([-0.5, -0.05, -0.01, 0.0, 0.01, 0.05, 0.9])
    f = analyze.sentiment_flags(c, SentimentSettings())
    assert f["sentiment_positive"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert f["sentiment_negative"].tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert f["sentiment_neutral"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert (f["sentiment_neutralpositive"] == (f["sentiment_neutral"] | f["sentiment_positive"])).all()
    v21 = analyze.sentiment_flags(c, SentimentSettings(positive_threshold=0, negative_threshold=0))
    assert v21["sentiment_positive"].tolist() == (c > 0).tolist()
    assert v21["sentiment_neutral"].tolist() == (c == 0).tolist()
    assert v21["sentiment_negative"].tolist() == (c < 0).tolist()


@pytest.fixture
def small_fw(tmp_path):
    sheets = framework_sheets(
        taxonomy={"risk": ["risk*"], "flood": ["flood*"], "ipcc": ["ipcc"]},
        co=[("co_rf", "risk", "2", "flood")],
        dc=[("doc_ipcc", "ipcc")],
        queries=[("q1", '"co_rf"'), ("q2", '"q1" and "doc_ipcc"'), ("q3", '"flood" and "sentiment_negative"')],
        variables=[("risk", "1", '"q1"'), ("ipcc risk", "1.1", '"q2"'), ("bad floods", "2", '"q3"')],
    )
    return load_framework(write_ods(tmp_path / "fw.ods", sheets))


def test_analysis_and_document_tables(small_fw):
    sents = pd.DataFrame({
        "document": ["d1", "d1", "d2", "d2"],
        "text": ["Flood risk rises.", "The IPCC agrees.", "Risk of terrible flooding.", "Calm day."],
        "vader_compound": [0.2, 0.0, -0.6, 0.3],
    })
    res = analyze.analyze_sentences(sents, small_fw, SentimentSettings())
    assert res["co_rf"].tolist() == [True, False, True, False]
    assert res["doc_ipcc"].tolist() == [True, True, False, False]
    assert res["q2"].tolist() == [True, False, False, False]  # query chaining
    assert res["sentences"].tolist()[0] == "flood risk rises."
    docs = analyze.document_outputs(res, small_fw, sentiment=True)
    rd = docs["results_document"].set_index("document")
    assert rd.loc["d1"].tolist() == [1, 1, 0] and rd.loc["d2"].tolist() == [1, 0, 1]
    pct = docs["results_document_percentage_of_sentences"].set_index("document")
    assert pct.loc["d1", "1. risk"] == 50.0
    sent = docs["results_document_sentiment"].set_index("document")
    assert sent.loc["d2", "2. bad floods"] == -0.6
    assert np.isnan(sent.loc["d1", "2. bad floods"])  # variable never fired in d1 -> empty cell


def test_long_segments_are_reported(small_fw, caplog):
    sents = pd.DataFrame({"document": ["d"], "text": ["word " * 600], "vader_compound": [0.0]})
    with caplog.at_level(logging.WARNING, logger="ctrlfindr"):
        analyze.analyze_sentences(sents, small_fw, SentimentSettings(), long_segment_words=500)
    assert "longer than 500 words" in caplog.text


def test_run_requires_ingest(tmp_path, small_fw):
    cfg = config_from_dict({"paths": {"framework": str(small_fw.path)}}, base=tmp_path)
    with pytest.raises(CtrlFindrError, match="run the ingest step first"):
        analyze.run(cfg, show_progress=False)


def test_run_keeps_source_tables_and_writes_lf(tmp_path, small_fw):
    (tmp_path / "Text_files").mkdir()
    (tmp_path / "Text_files" / "d1.txt").write_text("Flood risk rises. The IPCC agrees.", encoding="utf-8")
    cfg = config_from_dict({"paths": {"framework": str(small_fw.path)}}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)  # a second run must still find its input (v2.1 overwrote it)
    with sqlite3.connect(cfg.paths.database) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"documents", "sentences", "results_sentences"} <= tables
        assert conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 2
    raw = (cfg.paths.output / "results_document.tsv").read_bytes()
    assert b"\r\n" not in raw
    assert (cfg.paths.output / "run_metadata.json").is_file()
