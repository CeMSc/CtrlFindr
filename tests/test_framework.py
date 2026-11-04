import pytest

from ctrlfindr.common import ValidationError
from ctrlfindr.framework import load_framework
from fwutil import make_framework

TAX = [("fast", "flood*"), ("fast", "storm"), ("slow", "drought*"), ("hwc", "human~wildlife"),
       ("idioms", "business climate"), ("clim", "climat*"), ("clim", "-@idioms"), ("clim", "-climate of fear")]
QUERIES = [("a1", "fast AND NOT slow"), ("co_x", "NEAR(fast, slow, 5)"), ("a2", '"a1" OR co_x'),
           ("s", "slow AND sentiment_negative"), ("haz", "fast OR slow")]
VARIABLES = [("1", "Fast onset", "a1"), ("1.10", "Chained", "a2 OR s"), ("1,1", "Other", "V(1.10)")]


def build(tmp_path, ext="ods", **kw):
    args = dict(taxonomy=TAX, queries=QUERIES, variables=VARIABLES)
    args.update(kw)
    return make_framework(tmp_path / f"fw.{ext}", **args)


def problems(tmp_path, **kw):
    with pytest.raises(ValidationError) as exc:
        load_framework(build(tmp_path, **kw))
    return "\n".join(exc.value.problems)


@pytest.mark.parametrize("ext", ["ods", "xlsx"])
def test_valid_framework(tmp_path, ext):
    fw = load_framework(build(tmp_path, ext, meta={"title": "T", "my_key": "kept"}))
    assert list(fw.lists) == ["fast", "slow", "hwc", "idioms", "clim"]
    assert fw.lists["clim"].exclude_lists == ["idioms"] and fw.lists["clim"].exclude_terms == ["climate of fear"]
    assert fw.query_order.index("a1") < fw.query_order.index("a2")
    assert [v.column for v in fw.variables] == ["1. Fast onset", "1.10 Chained", "1.1 Other"]  # names keep case
    assert fw.variable("1.1").parent == "1" and fw.variable("1.10").parent == "1"
    assert fw.variable_order.index("1.10") < fw.variable_order.index("1.1")  # V(1.10) evaluated first
    assert fw.meta == {"format_version": "1", "title": "T", "my_key": "kept"}


def test_format_version_is_required_and_checked(tmp_path):
    assert "format_version' is missing" in problems(tmp_path, meta={"format_version": ""})
    assert "reads framework format 1" in problems(tmp_path, meta={"format_version": "2"})
    path = make_framework(tmp_path / "x.ods", TAX, QUERIES, VARIABLES, drop_sheets=("meta",))
    with pytest.raises(ValidationError, match="not a CtrlFindr framework"):
        load_framework(path)


def test_language(tmp_path):
    assert "not supported" in problems(tmp_path, meta={"language": "it"})


def test_settings(tmp_path):
    fw = load_framework(build(tmp_path, settings={"aggregate_children": "yes", "sentiment_positive_threshold": "0,1"}))
    assert fw.settings.aggregate_children and fw.settings.sentiment_positive_threshold == 0.1
    text = problems(tmp_path, settings={"sentiment": "maybe", "colour": "red", "sentiment_negative_threshold": "0.5"})
    assert "sentiment must be true or false" in text and "unknown setting 'colour'" in text and "thresholds must satisfy" in text


def test_sentiment_flags_need_sentiment(tmp_path):
    assert "needs sentiment" in problems(tmp_path, settings={"sentiment": False})


def test_name_rules(tmp_path):
    text = problems(tmp_path, taxonomy=TAX + [("fast onset", "x"), ("near", "y"), ("2x", "z"), ("sentiment_positive", "w")])
    assert "'fast onset' must start with a letter" in text and "'fast_onset'" in text
    assert "'near' is a reserved word" in text
    assert "'2x' must start with a letter" in text and "'x_2x'" in text
    assert "'sentiment_positive' is reserved by CtrlFindr" in text


def test_names_unique_across_lists_and_queries(tmp_path):
    assert "name 'fast' is already used in sheet 'taxonomy'" in problems(tmp_path, queries=QUERIES + [("fast", "slow")])


def test_taxonomy_problems(tmp_path):
    text = problems(tmp_path, taxonomy=TAX + [("bad", "~x"), ("bad2", "_ risk"), ("only_excl", "-foo"),
                                              ("x", "-@missing"), ("self", "a"), ("self", "-@self")])
    assert "'~' must join two parts" in text
    assert "word gap '_' needs words on both sides" in text
    assert "list 'only_excl'" in text and "only exclusions" in text
    assert "'-@missing' refers to a list that does not exist" in text
    assert "cannot exclude itself" in text


def test_exclusion_loop(tmp_path):
    text = problems(tmp_path, taxonomy=TAX + [("p", "a"), ("p", "-@q"), ("q", "b"), ("q", "-@p")])
    assert "exclude each other in a loop" in text


def test_query_problems(tmp_path):
    text = problems(tmp_path, queries=QUERIES + [
        ("q1", "fast AND missing"), ("q2", "NEAR(a1 AND slow, fast, 3)"), ("q3", "fast AND"),
        ("q4", "V(1)"), ("q5", "{-foo} OR fast"), ("q6", "COUNT(s) >= 2"),
    ])
    assert "'missing' is not a list, query or sentiment flag" in text
    assert "NEAR needs word positions" in text
    assert "query 'q3': the expression ends" in text
    assert "V(1) can only be used in the variables sheet" in text
    assert "exclusions belong in the taxonomy sheet" in text
    assert "COUNT needs word positions, but 's'" in text


def test_query_defined_as_or_of_lists_is_a_set(tmp_path):
    fw = load_framework(build(tmp_path, queries=QUERIES + [("q", "COUNT_DISTINCT(haz) >= 2"), ("n", "NEAR(haz, {rain}, 2)")]))
    assert sorted(set(x for v in fw.set_lists.values() for x in v)) == ["fast", "slow", "{rain}"]
    assert "{rain}" in fw.inline


def test_query_cycles(tmp_path):
    assert "loop: p -> r -> p" in problems(tmp_path, queries=QUERIES + [("p", "r"), ("r", "p")])


def test_variable_problems(tmp_path):
    text = problems(tmp_path, variables=[("1", "A", "a1"), ("1.", "Dup", "a1"), ("x.1", "Bad", "a1"), ("2", "Empty", ""),
                                         ("3", "Loop", "V(4)"), ("4", "Loop2", "V(3)"), ("5", "Ref", "V(9)"),
                                         ("6", "Parent", "a1", "8")])
    assert "number '1' is already used" in text
    assert "number 'x.1' must look like" in text
    assert "variable 2 has no definition" in text
    assert "variables refer to each other in a loop" in text
    assert "V(9) refers to a variable number that does not exist" in text
    assert "parent '8' is not a variable number" in text


def test_aggregate_children_allows_empty_parent(tmp_path):
    fw = load_framework(build(tmp_path, settings={"aggregate_children": True},
                              variables=[("1", "Parent", ""), ("1.1", "A", "a1"), ("1.2", "B", "s")]))
    assert fw.variable("1").children == ["1.1", "1.2"]
    assert fw.variable_order.index("1") > fw.variable_order.index("1.2")


def test_explicit_parent_overrides_numbering(tmp_path):
    fw = load_framework(build(tmp_path, variables=[("1", "A", "a1"), ("2", "B", "s"), ("2.1", "C", "a2", "1")]))
    assert fw.variable("2.1").parent == "1" and fw.variable("1").children == ["2.1"]


def test_tests_sheet_parsing(tmp_path):
    fw = load_framework(build(tmp_path, tests=[("Floods came.", "fast, a1, 1", "slow", "g1")]))
    t = fw.tests[0]
    assert (t.expect_true, t.expect_false, t.group) == (("fast", "a1", "1"), ("slow",), "g1")
    text = problems(tmp_path, tests=[("Floods.", "fast, nothing", ""), ("No expectations.", "", ""),
                                     ("Context only.", "", "", "g1")])
    assert "names 'nothing'" in text and "row 3: the test has no expect_true" in text and "row 4" not in text


def test_all_problems_reported_together(tmp_path):
    text = problems(tmp_path, taxonomy=TAX + [("bad", "~x")], queries=QUERIES + [("q", "nope")],
                    variables=VARIABLES + [("9", "X", "")])
    assert text.count("\n") >= 2


def test_sentiment_is_off_by_default(tmp_path):
    from ctrlfindr.spreadsheet import SheetSpec, write_workbook

    path = make_framework(tmp_path / "fw.ods", TAX, [("q", "fast")], [("1", "V", "q")])
    sheets = {"meta": [["key", "value"], ["format_version", "1"]], "taxonomy": [["list", "term"], ["a", "x"]],
              "queries": [["name", "query"], ["q", "a"]], "variables": [["number", "name", "definition"], ["1", "V", "q"]]}
    path = write_workbook(tmp_path / "nosettings.ods", [SheetSpec(k, v) for k, v in sheets.items()])
    assert load_framework(path).settings.sentiment is False


def test_exclude_sentence_types(tmp_path):
    fw = load_framework(build(tmp_path, settings={"exclude_sentence_types": "Header; table"}))
    assert fw.settings.exclude_sentence_types == ("header", "table")
    assert "unknown sentence type(s) 'footer'" in problems(tmp_path, settings={"exclude_sentence_types": "header, footer"})


def test_empty_setting_value_keeps_default(tmp_path):
    fw = load_framework(build(tmp_path, settings={"sentiment_positive_threshold": ""}))
    assert fw.settings.sentiment_positive_threshold == 0.05


def test_sentiment_without_vader_is_reported(tmp_path, monkeypatch):
    from ctrlfindr import sentiment

    monkeypatch.setattr(sentiment, "available", lambda: False)
    assert "needs the vaderSentiment package" in problems(tmp_path)
