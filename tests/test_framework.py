import pytest

from ctrlfindr.common import ValidationError
from ctrlfindr.framework import load_framework
from odsutil import framework_sheets, write_ods

TAX = {"fast": ["flood*", "storm"], "slow": ["drought*"], "hwc": ["human–wildlife"]}
OK = dict(
    taxonomy=TAX,
    co=[("co_x", "fast", (5, "5"), "slow")],
    dc=[("doc_hwc", "hwc")],
    queries=[("a1", '"fast" AND NOT("slow")'), ("a2", '"a1" or "co_x"'), ("s", '"slow" and "sentiment_negative"')],
    variables=[("fast onset", "1", '"a1"'), ("chained", "1.10", '"a2", "s"'), ("other", ("1.1", "1.1"), '"a1"')],
)


def build(tmp_path, **overrides):
    args = {**OK, **overrides}
    return write_ods(tmp_path / "fw.ods", framework_sheets(**args))


def problems(tmp_path, sentiment=True, **overrides):
    with pytest.raises(ValidationError) as exc:
        load_framework(build(tmp_path, **overrides), sentiment_enabled=sentiment)
    return "\n".join(exc.value.problems)


def test_valid_framework(tmp_path):
    fw = load_framework(build(tmp_path))
    assert list(fw.taxonomy) == ["fast", "slow", "hwc"]
    assert fw.taxonomy["hwc"] == ["human-wildlife"]  # en dash folded, as in the corpus
    assert fw.co_occurrences[0].distance == 5
    assert [v.column for v in fw.variables] == ["1. fast onset", "1.10 chained", "1.1 other"]  # 1.10 stays 1.10
    assert fw.query_order.index("a1") < fw.query_order.index("a2")
    assert fw.uses_sentiment


def test_example_framework_loads(example_framework_path):
    fw = load_framework(example_framework_path)
    assert len(fw.taxonomy) == 15 and len(fw.variables) == 11
    assert [d.name for d in fw.doc_conditionals] == ["doc_ipcc"]


def test_all_problems_are_reported_together(tmp_path):
    text = problems(
        tmp_path,
        taxonomy={"fast": ["flood"], "empty": [], "slow": ["*"]},
        co=[("", "fast", "5", "slow"), ("co_bad", "fast", "x", "nolist")],
        queries=[("a1", '"fast" and "missing"'), ("a2", '"fast" and')],
        variables=[("v", "1", '"a1" OR "a2"')],
    )
    for expected in [
        "list 'empty' has no terms",
        "term '*' has no letters",
        "empty cell(s) in 'name of co-occurrence'",
        "distance 'x' must be a whole number",
        "second list 'nolist' is not a list",
        "refers to 'missing'",
        "query 'a2': query ends",
        "must be query names in double quotes separated by commas",
    ]:
        assert expected in text, expected


def test_names_must_be_unique_across_sheets(tmp_path):
    text = problems(tmp_path, queries=[("fast", '"slow"'), ("a1", '"fast"'), ("a2", '"a1"'), ("s", '"slow"')])
    assert "name 'fast' is already used in sheet 'taxonomy'" in text


def test_reserved_names(tmp_path):
    text = problems(tmp_path, taxonomy={**TAX, "sentiment_positive": ["good"]})
    assert "'sentiment_positive' is reserved" in text


def test_duplicate_variable_numbers(tmp_path):
    text = problems(tmp_path, variables=[("x", "1", '"a1"'), ("y", "1.", '"a1"')])
    assert "variable number '1.' is already used" in text


def test_query_cycles(tmp_path):
    text = problems(tmp_path, queries=[("a1", '"a2"'), ("a2", '"a1"'), ("s", '"slow"')],
                    variables=[("v", "1", '"a1"')])
    assert "loop: a1 -> a2 -> a1" in text


def test_sentiment_terms_need_sentiment_enabled(tmp_path):
    assert "sentiment analysis is disabled" in problems(tmp_path, sentiment=False)


def test_missing_sheet_and_file(tmp_path):
    sheets = framework_sheets(**OK)
    del sheets["taxonomy"]
    with pytest.raises(ValidationError, match="invalid"):
        load_framework(write_ods(tmp_path / "x.ods", sheets))
    with pytest.raises(ValidationError, match="not found"):
        load_framework(tmp_path / "nope.ods")


def test_curly_quotes_in_queries_are_accepted(tmp_path):
    fw = load_framework(build(tmp_path, queries=[("a1", "“fast” AND NOT(“slow”)"), ("a2", '"a1"'), ("s", '"slow"')]))
    assert fw.queries[0].name == "a1"
