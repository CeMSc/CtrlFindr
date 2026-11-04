"""Regression tests against committed outputs.

* v2.1: the 2.1 code (legacy/) must reproduce its own committed outputs, and the 3.0 engine
  with the example framework (format 1, the same rules rewritten) must, on the sentences
  committed with 2.1 and with 2.1's sentiment scores and thresholds of 0, change exactly the
  rows listed and explained in fixtures/v21_sentences_expected_changes.json.
* v3: the full pipeline on examples/example_project must reproduce fixtures/golden_v3.
"""

import importlib.util
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import EXAMPLE, FIXTURES, ROOT
from ctrlfindr import analyze, config_from_dict, ingest
from ctrlfindr.analyze import OUTPUT_FILES, document_outputs
from ctrlfindr.common import normalize_text
from ctrlfindr.engine import analyze_sentences
from ctrlfindr.framework import load_framework

V21 = FIXTURES / "golden_v2.1"
V3 = FIXTURES / "golden_v3"


def assert_frames_equal(new: pd.DataFrame, old: pd.DataFrame, name: str, skip=()):
    assert list(new.columns) == list(old.columns), name
    assert len(new) == len(old), name
    for col in old.columns:
        if col in skip:
            continue
        a, b = new[col], old[col]
        if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
            ok = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            ok = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        assert ok.all(), f"{name}: column {col!r} differs in {int((~ok).sum())} rows, first at {int(np.flatnonzero(~ok)[0])}"


def test_v21_fixtures_are_reproducible(golden_v21_sentences):
    """The extracted 2.1 code reproduces its own committed outputs (fixture sanity check)."""
    pytest.importorskip("odf", reason="the 2.1 code reads its framework with odfpy")
    spec = importlib.util.spec_from_file_location("ctrlfindr_v21", ROOT / "legacy" / "v2.1" / "ctrlfindr_v21.py")
    v21 = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(v21)
    out = v21.run_analysis_from_sentences(golden_v21_sentences, str(V21 / "Assessment_framework_v2.1.ods"),
                                          compound=golden_v21_sentences["vadar_compound"])
    for name in OUTPUT_FILES:
        assert_frames_equal(out[name].reset_index(drop=True), pd.read_csv(V21 / f"{name}.tsv", sep="\t"), name)


@pytest.fixture(scope="module")
def v3_on_v21_sentences(golden_v21_sentences):
    fw = load_framework(EXAMPLE / "Input" / "Assessment_framework.ods")
    fw.settings.sentiment_positive_threshold = fw.settings.sentiment_negative_threshold = 0  # as in 2.1
    g = golden_v21_sentences
    inp = pd.DataFrame({"document": g["document"], "text": g["sentences"], "vader_compound": g["vadar_compound"]})
    res = analyze_sentences(inp, fw)
    return fw, res


def lower_columns(df):
    return df.rename(columns=str.lower)  # 2.1 lowercased variable names


def test_v3_engine_on_v21_sentences_changes_only_reviewed_rows(v3_on_v21_sentences, golden_v21_sentences):
    _, res = v3_on_v21_sentences
    res = lower_columns(res)
    old = golden_v21_sentences.rename(columns={"vadar_compound": "vader_compound"})
    expected = json.loads((FIXTURES / "v21_sentences_expected_changes.json").read_text(encoding="utf-8"))
    assert set(res.columns) == set(old.columns)
    assert (res["sentences"] == old["sentences"].map(lambda s: normalize_text(s).lower())).all()
    for col in old.columns:
        if old[col].dtype != bool:
            continue
        diff = np.flatnonzero(res[col].to_numpy() != old[col].to_numpy()).tolist()
        assert diff == expected.get(col, {}).get("rows", []), col


def test_v3_document_tables_on_v21_sentences(v3_on_v21_sentences):
    fw, res = v3_on_v21_sentences
    expected = json.loads((FIXTURES / "v21_sentences_expected_changes.json").read_text(encoding="utf-8"))
    changed = {c for c in expected}
    for name, df in document_outputs(res, fw).items():
        old = pd.read_csv(V21 / f"{name}.tsv", sep="\t")
        assert_frames_equal(lower_columns(df), old, name, skip=changed)


def test_example_framework_tests_pass():
    from ctrlfindr.engine import run_tests

    fw = load_framework(EXAMPLE / "Input" / "Assessment_framework.ods")
    assert len(fw.tests) >= 5 and run_tests(fw) == []


def test_end_to_end_example_matches_golden_v3(tmp_path):
    shutil.copytree(EXAMPLE / "Text_files", tmp_path / "Text_files")
    shutil.copytree(EXAMPLE / "Input", tmp_path / "Input")
    cfg = config_from_dict({}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    for name in OUTPUT_FILES:
        new = pd.read_csv(cfg.paths.output / f"{name}.tsv", sep="\t")
        assert_frames_equal(new, pd.read_csv(V3 / f"{name}.tsv", sep="\t"), name)


def test_markdown_example(tmp_path):
    from conftest import ROOT as root

    src = root / "examples" / "markdown_project"
    shutil.copytree(src / "Text_files", tmp_path / "Text_files")
    shutil.copytree(src / "Input", tmp_path / "Input")
    cfg = config_from_dict({"paths": {"framework": "Input/Assessment_framework.xlsx"}}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    res = pd.read_csv(cfg.paths.output / "results_sentences.tsv", sep="\t")
    assert set(res["structure"]) == {"body", "list"}  # headers and table rows excluded by the framework
    assert len(res) == 14 and res["1. Drought-related conflict"].sum() >= 2
