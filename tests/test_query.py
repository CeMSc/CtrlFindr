import pytest

from ctrlfindr import query as q


@pytest.mark.parametrize("text,canonical", [
    ('fast AND NOT (slow OR co_slow)', "fast AND NOT (slow OR co_slow)"),
    ('"fast" and not("slow")', "fast AND NOT slow"),                     # quotes optional, case-insensitive
    ('(("a")AND("b"))OR"c"', "(a AND b) OR c"),
    ('NEAR(fast OR {storm*}, hwc, 5)', "NEAR(fast OR {storm*}, hwc, 5)"),
    ('doc(ipcc) >= 3 and doc_share(a1)>=0.05', "DOC(ipcc) >= 3 AND DOC_SHARE(a1) >= 0.05"),
    ('doc(x)', "DOC(x)"), ('DOC(x) > 2', "DOC(x) > 2"), ('WINDOW(c, 2) AND d', "WINDOW(c, 2) AND d"),
    ('count_distinct(h) >= 2', "COUNT_DISTINCT(h) >= 2"), ('V(2.1) or V(2.2)', "V(2.1) OR V(2.2)"),
    ('not_near(a, b, 3)', "NOT_NEAR(a, b, 3)"), ('before(a, b, 0)', "BEFORE(a, b, 0)"),
    ('"command-and-control"', "command-and-control"),                  # operator words inside names stay
])
def test_parse_and_canonical_text(text, canonical):
    assert q.to_text(q.parse(text)) == canonical


def test_precedence_not_and_or():
    node = q.parse("a or b and not c")
    assert isinstance(node, q.Or) and isinstance(node.operands[1], q.And)
    assert isinstance(node.operands[1].operands[1], q.Not)


def test_function_name_without_parenthesis_is_a_name():
    assert q.parse("near") == q.Name("near")


def test_paragraph_is_not_a_function():
    assert q.parse("paragraph") == q.Name("paragraph")  # an ordinary name now
    with pytest.raises(q.QuerySyntaxError):
        q.parse("PARAGRAPH(x)")


def test_bare_count_means_at_least_one():
    c = q.parse("COUNT(x)")
    assert (c.op, c.value) == (">=", 1.0)
    assert (q.parse("DOC_SHARE(x)").op, q.parse("DOC_SHARE(x)").value) == (">", 0.0)


@pytest.mark.parametrize("bad,msg", [
    ("NEAR(a, b)", "takes 3 arguments"), ("NEAR(a, b, 2.5)", "whole number"), ("WINDOW(a, 2) > 1", "cannot be compared"),
    ("doc_share(a) > 2", "between 0 and 1"), ("fast AND", "ends where"), ("{storm", "unmatched '{'"),
    ("a & b", "unexpected character"), ("V(x)", "variable number"), ("a b", "unexpected 'b'"), ('""', "empty name"),
    ("{}", "empty term"), ('"a', "unclosed"), ("", "empty"), ("(a", r"missing '\)'"), ("COUNT(a) >=", "a number after"),
])
def test_syntax_errors(bad, msg):
    with pytest.raises(q.QuerySyntaxError, match=msg):
        q.parse(bad)


def test_helpers():
    node = q.parse("NEAR(a OR {x y}, b, 2) AND DOC(c) AND V(1.2)")
    assert q.names(node) == {"a", "b", "c"}
    assert q.inline_terms(node) == {"x y"}
    assert q.variable_refs(node) == {"1.2"}
    assert [(f, q.to_text(a)) for f, a in q.set_arguments(node)] == [("near", "a OR {x y}"), ("near", "b")]


def test_name_rules():
    assert q.check_name("co_slow") is None and q.check_name("doc-ipcc") is None
    assert "reserved word" in q.check_name("window")
    assert "'fast_onset'" in q.check_name("fast onset")


def test_dependency_order():
    order, cycles = q.dependency_order({"q1": {"q2"}, "q2": set(), "q3": {"q1"}})
    assert order == ["q2", "q1", "q3"] and cycles == []
    assert q.dependency_order({"a": {"b"}, "b": {"a"}})[1] == [["a", "b", "a"]]
