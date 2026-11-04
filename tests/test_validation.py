import pandas as pd
import pytest

from ctrlfindr import analyze, config_from_dict, ingest
from ctrlfindr.common import CtrlFindrError, ValidationError
from ctrlfindr.validation import sample
from fwutil import make_framework


@pytest.fixture
def analyzed(tmp_path):
    texts = ["Floods hit the town.", "Drought hurt farmers.", "Calm weather.", "The market rose.",
             "Floods and storms again.", "Nothing to report.", "Storm damage everywhere.", "A quiet day."]
    (tmp_path / "Text_files").mkdir()
    (tmp_path / "Text_files" / "d.txt").write_text(" ".join(texts), encoding="utf-8")
    fw = make_framework(tmp_path / "fw.ods", [("haz", "flood*"), ("haz", "storm*")], [("h", "haz")], [("1", "Hazard", "h")])
    cfg = config_from_dict({"paths": {"framework": str(fw)}}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    return cfg


def test_sample_is_stratified_and_blind(analyzed):
    path = sample(analyzed, n=2, seed=3)
    s = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(s.columns) == ["document", "sentence_id", "variable", "sentence", "human_code", "note"]
    assert len(s) == 4 and set(s["variable"]) == {"1"}
    assert sorted(s["sentence"].str.contains("Flood|Storm").tolist()) == [False, False, True, True]
    assert (s["human_code"] == "").all()


def test_report_precision_and_estimated_recall(analyzed):
    path = sample(analyzed, n=3, seed=1)
    s = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # the coder also counts drought as a hazard; CtrlFindr's list does not
    s["human_code"] = s["sentence"].str.contains("Flood|Storm|Drought").astype(int).astype(str)
    s.loc[s["sentence"].str.startswith("Storm damage"), "human_code"] = "0"  # one false positive
    analyzed.paths.validation.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(analyzed.paths.validation, sep="\t", index=False)
    analyze.run(analyzed, show_progress=False)
    rep = pd.read_csv(analyzed.paths.output / "validation_report.tsv", sep="\t").iloc[0]
    assert (rep.tp, rep.fp, rep.coded_flagged) == (2, 1, 3)
    assert rep.precision == pytest.approx(2 / 3)
    # FN rate in the unflagged sample (1 of 3) scaled to the 5 unflagged sentences in the corpus
    tp_est, fn_est = (2 / 3) * 3, (1 / 3) * 5
    assert rep.recall_estimated == pytest.approx(tp_est / (tp_est + fn_est))
    dis = pd.read_csv(analyzed.paths.output / "validation_disagreements.tsv", sep="\t")
    assert len(dis) == 2


def test_invalid_codes_are_reported(analyzed):
    analyzed.paths.validation.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"document": ["d"], "sentence_id": [1], "variable": ["1"], "human_code": ["maybe"]}).to_csv(
        analyzed.paths.validation, sep="\t", index=False)
    with pytest.raises(ValidationError, match="invalid codes"):
        analyze.run(analyzed, show_progress=False)


def test_codes_follow_their_sentence_after_the_document_changes(tmp_path, caplog):
    import logging

    (tmp_path / "Text_files").mkdir()
    f = tmp_path / "Text_files" / "a.txt"
    f.write_text("Floods hit. Calm day. Storms came. Quiet. Floods again.", encoding="utf-8")
    fw = make_framework(tmp_path / "fw.ods", [("h", "flood*"), ("h", "storm*")], [("q", "h")], [("1", "H", "q")],
                        settings={"sentiment": False})
    cfg = config_from_dict({"paths": {"framework": str(fw)}}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    s = pd.read_csv(sample(cfg, n=2, seed=1), sep="\t", dtype=str, keep_default_na=False)
    s["human_code"] = s["sentence"].str.contains("Flood|Storm").astype(int).astype(str)  # agrees with CtrlFindr
    cfg.paths.validation.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(cfg.paths.validation, sep="\t", index=False)
    # one sentence inserted at the start and one coded sentence deleted: all numbers shift
    f.write_text("Introduction. Floods hit. Calm day. Storms came. Floods again.", encoding="utf-8")
    ingest.run(cfg, show_progress=False)
    with caplog.at_level(logging.INFO, logger="ctrlfindr"):
        analyze.run(cfg, show_progress=False)
    rep = pd.read_csv(cfg.paths.output / "validation_report.tsv", sep="\t").iloc[0]
    assert rep.fp == 0 and rep.fn == 0  # moved sentences are found again, nothing is misaligned
    assert rep.coded == len(s) - int(s["sentence"].eq("Quiet.").sum())
    assert "found under a new sentence number" in caplog.text
    if s["sentence"].eq("Quiet.").any():
        assert "left out because the coded sentence is no longer in its document" in caplog.text


def test_sample_refuses_results_older_than_the_sentences(analyzed):
    (analyzed.paths.text_files / "d.txt").write_text("Floods. Changed text.", encoding="utf-8")
    ingest.run(analyzed, show_progress=False)
    with pytest.raises(CtrlFindrError, match="run the analysis first"):
        sample(analyzed, n=2)
