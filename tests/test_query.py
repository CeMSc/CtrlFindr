import numpy as np
import pytest

from ctrlfindr import query as q

T, F = True, False
COLS = {"a": np.array([T, T, F, F]), "b": np.array([T, F, T, F]), "c": np.array([F, F, F, T]),
        "command-and-control": np.array([T, F, F, F])}


def ev(text):
    return q.evaluate(q.parse(text), COLS).tolist()


def test_operators_case_insensitive_and_adjacent_parentheses():
    assert ev('"a" AND NOT("b")') == ev('"a" and not ("b")') == [F, T, F, F]
    assert ev('(("a")AND("b"))OR"c"') == [T, F, F, T]


@pytest.mark.parametrize("expr", ['"a" or "b" and "c"', 'not "a" and "b"', 'not "a" or "b" and not "c"', '"a" and "b" or "c"'])
def test_precedence_matches_python(expr):
    """v2.x evaluated queries with Python's eval; precedence must be identical."""
    expected = []
    for i in range(4):
        env = {k: bool(v[i]) for k, v in COLS.items() if k in "abc"}
        py = expr.replace('"', "")
        expected.append(bool(eval(py, {}, env)))
    assert ev(expr) == expected


def test_operator_words_inside_terms_are_not_rewritten():
    assert ev('"command-and-control"') == [T, F, F, F]


def test_unknown_term_raises_instead_of_being_truthy():
    with pytest.raises(KeyError):
        ev('"sentiment_positive" and "a"')


@pytest.mark.parametrize("bad,msg", [
    ('"a" and', "ends"), ('("a"', r"missing '\)'"), ('"a")', "unexpected"), ('a and "b"', "unexpected word"),
    ('"a" "b"', "unexpected"), ('""', "empty term"), ('"a', "unclosed"), ("", "empty"), ('"a" & "b"', "unexpected character"),
])
def test_syntax_errors(bad, msg):
    with pytest.raises(q.QuerySyntaxError, match=msg):
        q.parse(bad)


def test_terms_and_dependency_order():
    assert q.terms(q.parse('("x" or not "y") and "z"')) == {"x", "y", "z"}
    order, cycles = q.dependency_order({"q1": {"q2"}, "q2": set(), "q3": {"q1"}})
    assert order == ["q2", "q1", "q3"] and cycles == []
    _, cycles = q.dependency_order({"q1": {"q2"}, "q2": {"q1"}})
    assert cycles == [["q1", "q2", "q1"]]
