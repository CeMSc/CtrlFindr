"""Regression tests against committed outputs.

* v2.1: the analysis step is run on the sentences committed with CtrlFindr 2.1 (with the
  2.1 sentiment scores and thresholds of 0) and must match the 2.1 outputs exactly, except
  for the rows listed and explained in fixtures/v21_sentences_expected_changes.json
  (see docs/result_changes.md). This isolates the analysis from sentence splitting.
* v3: the full pipeline on examples/example_project must reproduce fixtures/golden_v3.
"""

import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from conftest import EXAMPLE, FIXTURES, ROOT
from ctrlfindr import analyze, config_from_dict, ingest
from ctrlfindr.analyze import OUTPUT_FILES, analyze_sentences, document_outputs
from ctrlfindr.common import normalize_text
from ctrlfindr.config import SentimentSettings
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
            ok = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
        assert ok.all(), f"{name}: column {col!r} differs in {int((~ok).sum())} rows, first at {int(np.flatnonzero(~ok)[0])}"


def load_v21_module():
    spec = importlib.util.spec_from_file_location("ctrlfindr_v21", ROOT / "legacy" / "v2.1" / "ctrlfindr_v21.py")
    mod = importlib.util.module_from_spec(spec)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # v2.1 has an invalid escape sequence
        spec.loader.exec_module(mod)
    return mod


def test_v21_fixtures_are_reproducible(golden_v21_sentences, example_framework_path):
    """The extracted v2.1 code reproduces its own committed outputs (fixture sanity check)."""
    pytest.importorskip("odf", reason="the v2.1 code reads the framework with odfpy")
    v21 = load_v21_module()
    out = v21.run_analysis_from_sentences(golden_v21_sentences, str(example_framework_path),
                                          compound=golden_v21_sentences["vadar_compound"])
    for name in OUTPUT_FILES:
        assert_frames_equal(out[name].reset_index(drop=True), pd.read_csv(V21 / f"{name}.tsv", sep="\t"), name)


@pytest.fixture(scope="module")
def v3_on_v21_sentences(golden_v21_sentences, example_framework_path):
    fw = load_framework(example_framework_path)
    g = golden_v21_sentences
    inp = pd.DataFrame({"document": g["document"], "text": g["sentences"], "vader_compound": g["vadar_compound"]})
    res = analyze_sentences(inp, fw, SentimentSettings(positive_threshold=0, negative_threshold=0))
    return fw, res


def test_v3_analysis_on_v21_sentences_changes_only_reviewed_rows(v3_on_v21_sentences, golden_v21_sentences):
    _, res = v3_on_v21_sentences
    old = golden_v21_sentences.rename(columns={"vadar_compound": "vader_compound"})
    expected = json.loads((FIXTURES / "v21_sentences_expected_changes.json").read_text(encoding="utf-8"))
    assert list(res.columns) == list(old.columns)
    # Sentence text differs only by normalization (dashes, quotes, NFKC).
    assert (res["sentences"] == old["sentences"].map(lambda s: normalize_text(s).lower())).all()
    for col in old.columns:
        if old[col].dtype != bool:
            continue
        diff = np.flatnonzero(res[col].to_numpy() != old[col].to_numpy()).tolist()
        assert diff == expected.get(col, {}).get("rows", []), col


def test_v3_document_tables_on_v21_sentences_follow_sentence_changes(v3_on_v21_sentences, golden_v21_sentences):
    fw, res = v3_on_v21_sentences
    expected = json.loads((FIXTURES / "v21_sentences_expected_changes.json").read_text(encoding="utf-8"))
    changed_vars = {v.column for v in fw.variables if v.column in expected}
    docs = document_outputs(res, fw, sentiment=True)
    for name, df in docs.items():
        old = pd.read_csv(V21 / f"{name}.tsv", sep="\t")
        assert_frames_equal(df, old, name, skip=changed_vars)


def test_end_to_end_example_matches_golden_v3(tmp_path):
    shutil.copytree(EXAMPLE / "Text_files", tmp_path / "Text_files")
    shutil.copytree(EXAMPLE / "Input", tmp_path / "Input")
    cfg = config_from_dict({}, base=tmp_path)
    ingest.run(cfg, show_progress=False)
    analyze.run(cfg, show_progress=False)
    for name in OUTPUT_FILES:
        new = pd.read_csv(cfg.paths.output / f"{name}.tsv", sep="\t")
        assert_frames_equal(new, pd.read_csv(V3 / f"{name}.tsv", sep="\t"), name)
