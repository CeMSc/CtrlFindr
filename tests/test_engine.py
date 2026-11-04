import logging

import numpy as np
import pandas as pd
import pytest

from ctrlfindr.engine import analyze_sentences, run_tests, sentiment_flags
from ctrlfindr.framework import Settings, load_framework
from fwutil import make_framework

TAX = [("haz", "flood*"), ("haz", "drought*"), ("haz", "storm*"), ("conflict", "conflict*"), ("conflict", "clash*"),
       ("ipcc", "ipcc"), ("clim", "climat*"), ("clim", "-business climate"), ("wild", "wildlife")]

# document d1: 3 sentences, 2 paragraphs; document d2: 2 sentences
SENTENCES = pd.DataFrame({
    "document": ["d1", "d1", "d1", "d2", "d2"],
    "paragraph": [1, 1, 2, 1, 1],
    "text": [
        "Floods and droughts caused conflict.",                       # 0
        "The IPCC warned about the business climate.",                # 1
        "Calm years followed.",                                       # 2
        "Storms, more storms, then a long drought and clashes.",       # 3
        "Conflict over wildlife followed the climate shift.",          # 4
    ],
    "vader_compound": [-0.6, 0.0, 0.3, -0.2, 0.04],
})


def run(tmp_path, queries, variables=None, settings=None, taxonomy=TAX, sentences=SENTENCES):
    names = [qq[0] for qq in queries]
    variables = variables or [("1", "All", " OR ".join(names))]
    fw = load_framework(make_framework(tmp_path / "fw.ods", taxonomy, queries, variables, settings))
    return analyze_sentences(sentences, fw), fw


def col(res, name):
    return res[name].tolist()


def test_lists_inline_terms_and_exclusions(tmp_path):
    res, _ = run(tmp_path, [("q1", "haz AND {caus*}"), ("q2", "clim")])
    assert col(res, "q1") == [True, False, False, False, False]
    assert col(res, "q2") == [False, False, False, False, True]  # "business climate" is excluded


@pytest.mark.parametrize("query,expected", [
    ("NEAR(haz, conflict, 0)", [False, False, False, False, False]),
    ("NEAR(haz, conflict, 1)", [True, False, False, True, False]),   # "droughts caused conflict", "drought and clashes"
    ("BEFORE(conflict, haz, 5)", [False, False, False, False, False]),
    ("BEFORE(haz, conflict, 5)", [True, False, False, True, False]),
    ("NOT_NEAR(haz, conflict, 2)", [True, False, False, True, False]),  # "floods" is 3 words from "conflict"
    ("NOT_NEAR(haz, conflict, 3)", [False, False, False, True, False]),  # "storms" far from "clashes"
    ("NEAR(haz OR {calm}, wild OR conflict, 10)", [True, False, False, True, False]),
])
def test_proximity(tmp_path, query, expected):
    res, _ = run(tmp_path, [("q", query)])
    assert col(res, "q") == expected


@pytest.mark.parametrize("query,expected", [
    ("DOC(ipcc)", [True, True, True, False, False]),
    ("DOC(haz) >= 2", [False, False, False, False, False]),
    ("DOC(conflict) >= 2", [False, False, False, True, True]),
    ("DOC_SHARE(haz) >= 0.5", [False, False, False, True, True]),
    ("DOC_SHARE(haz) > 0.3", [True, True, True, True, True]),
    ("PARAGRAPH(ipcc)", [True, True, False, False, False]),
    ("PARAGRAPH(haz) = 0", [False, False, True, False, False]),
    ("WINDOW(ipcc, 1)", [True, True, True, False, False]),
    ("WINDOW(haz, 1) AND NOT haz", [False, True, False, False, True]),  # windows stop at document boundaries
    ("COUNT(haz) >= 3", [False, False, False, True, False]),
    ("COUNT_DISTINCT(haz) >= 2", [True, False, False, True, False]),
    ("COUNT_DISTINCT(haz OR conflict) = 3", [True, False, False, True, False]),
    ("COUNT(haz) < 1", [False, True, True, False, True]),
])
def test_scope_and_count_functions(tmp_path, query, expected):
    res, _ = run(tmp_path, [("q", query)])
    assert col(res, "q") == expected


def test_sentiment_settings_come_from_the_framework(tmp_path):
    res, _ = run(tmp_path, [("q", "sentiment_positive")], settings={"sentiment_positive_threshold": 0.3})
    assert col(res, "q") == [False, False, True, False, False]
    res, _ = run(tmp_path, [("q", "haz")], settings={"sentiment": False})
    assert "vader_compound" not in res.columns and "sentiment_positive" not in res.columns


def test_sentiment_flags():
    c = np.array([-0.5, -0.05, -0.01, 0.0, 0.01, 0.05, 0.9])
    f = sentiment_flags(c, Settings())
    assert f["sentiment_positive"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert f["sentiment_negative"].tolist() == [1, 1, 0, 0, 0, 0, 0]
    zero = sentiment_flags(c, Settings(sentiment_positive_threshold=0, sentiment_negative_threshold=0))
    assert zero["sentiment_positive"].tolist() == (c > 0).tolist() and zero["sentiment_neutral"].tolist() == (c == 0).tolist()


def test_variables_with_references_and_aggregation(tmp_path):
    queries = [("h", "haz"), ("c", "conflict")]
    res, fw = run(tmp_path, queries, variables=[("1", "Both", "V(1.1) AND V(1.2)"), ("1.1", "Hazard", "h"),
                                                 ("1.2", "Conflict", "c")])
    assert col(res, "1. Both") == [True, False, False, True, False]
    res, _ = run(tmp_path, queries, settings={"aggregate_children": True},
                 variables=[("1", "Any", ""), ("1.1", "Hazard", "h"), ("1.2", "Conflict", "c")])
    assert col(res, "1. Any") == [True, False, False, True, True]


def test_hierarchy_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ctrlfindr"):
        run(tmp_path, [("h", "haz"), ("c", "conflict")], variables=[("1", "Parent", "h"), ("1.1", "Child", "c")])
    assert "true in sentences where their parent is false" in caplog.text and "1.1 vs parent 1: 1 sentences" in caplog.text


def test_output_columns(tmp_path):
    res, fw = run(tmp_path, [("q", "haz")])
    assert list(res.columns)[:3] == ["paragraph", "document", "sentences"]
    assert "{caus*}" not in res.columns  # inline terms are not exported
    assert list(res.columns)[-1] == "1. All"


def test_framework_tests_pass_and_fail(tmp_path):
    tests = [("Floods hit the coast.", "haz, q", "conflict"),
             ("The IPCC report.", "", "", "g"), ("Clashes erupted.", "d", "", "g"),
             ("Floods hit the coast.", "conflict", "haz")]
    fw = load_framework(make_framework(tmp_path / "fw.ods", TAX, [("q", "haz"), ("d", "conflict AND DOC(ipcc)")],
                                       [("1", "V", "q")], tests=tests))
    failures = run_tests(fw)
    assert len(failures) == 2 and all("tests row 5" in f for f in failures)
    assert "expected conflict to be true, got false" in failures[0]
    assert "Matched: haz: floods" in failures[0]
