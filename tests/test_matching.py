import re

import numpy as np
import pytest

from ctrlfindr.matching import Matcher, co_occurs, term_regex


def lists_found(taxonomy, sentence):
    m = Matcher(taxonomy)
    return {name for name, spans in zip(m.lists, m.spans(sentence)) if spans}


@pytest.mark.parametrize("term,sentence,hit", [
    ("flood", "the flood came", True),
    ("flood", "the floods came", False),          # whole words only
    ("flood*", "the floods came", True),
    ("flood*", "flash-flooding", True),           # hyphen is a boundary
    ("flash flood*", "a flash flooding event", True),
    ("climat*", "anticlimatic", False),
    ("u.s.", "the u.s. policy", True),            # term ending in punctuation (v2.1 never matched this)
    ("biodivers* loss", "biodiversity loss is", True),  # inner wildcard
    ("*fire", "a bushfire", True),                # leading wildcard
    ("co2", "co2 emissions", True),
    ("niño", "el niño year", True),               # non-ASCII letters are word characters
    ("niño", "el niños year", False),
    ("ipcc", "the ipcc_report", False),           # underscore is a word character, as in regex \w
])
def test_term_semantics(term, sentence, hit):
    assert bool(lists_found({"l": [term]}, sentence)) is hit
    assert bool(term_regex(term).search(sentence)) is hit  # automaton == regex definition


def test_automaton_equals_regex_alternation_on_random_terms():
    terms = ["climate change", "climat*", "warm*", "heat", "heat wave*", "sea level*", "rise", "drought*", "el niño"]
    rx = re.compile("|".join(term_regex(t).pattern for t in terms))
    sentences = ["the climate is changing", "heatwaves and heat waves", "sea levels rise", "warming", "el niño and la niña",
                 "nothing here", "climatechange", "a drought, droughts", "heat-wave"]
    m = Matcher({"l": terms})
    for s in sentences:
        assert bool(m.spans(s)[0]) == bool(rx.search(s)), s


def test_term_in_two_lists_and_spans():
    m = Matcher({"a": ["encroach*", "land"], "b": ["encroach*"]})
    spans = m.spans("encroachment on land")
    assert spans[0] == {(0, 12), (16, 20)} and spans[1] == {(0, 12)}
    hits, keep = m.match(["encroachment on land", "nothing"], keep_spans={"b"})
    assert hits["a"].tolist() == [True, False] and keep["b"] == [[(0, 12)], None]


def span(s, word, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = s.index(word, i + 1)
    return (i, i + len(word))


def test_distance_counts_words_strictly_between():
    s = "increase the adaptive capacity"
    a, b = [span(s, "increase")], [span(s, "capacity")]
    assert co_occurs(s, a, b, 2) and not co_occurs(s, a, b, 1)
    adj = "increase capacity"
    assert co_occurs(adj, [span(adj, "increase")], [span(adj, "capacity")], 0)


def test_distance_ignores_order_and_punctuation_tokens():
    s = "capacity - which we increase"
    assert co_occurs(s, [span(s, "increase")], [span(s, "capacity")], 2)  # "-" is not a word
    assert not co_occurs(s, [span(s, "increase")], [span(s, "capacity")], 1)


def test_same_word_in_both_lists_is_not_a_co_occurrence():
    """v2.1 bug: encroach* in both lists co-occurred with itself at distance 0."""
    s = "human encroachment continues"
    sp = [span(s, "encroachment")]
    assert not co_occurs(s, sp, sp, 5)
    m = Matcher({"x": ["encroach*"], "y": ["encroach*"]})
    _, keep = m.match([s], {"x", "y"})
    assert not co_occurs(s, keep["x"][0], keep["y"][0], 5)


def test_overlapping_matches_do_not_count():
    s = "climate change matters"
    assert not co_occurs(s, [span(s, "climate")], [(0, len("climate change"))], 3)


def test_nearest_pair_is_used():
    s = "increase " + "x " * 10 + "capacity and increase capacity"
    m = Matcher({"i": ["increase"], "c": ["capacity"]})
    _, keep = m.match([s], {"i", "c"})
    assert co_occurs(s, keep["i"][0], keep["c"][0], 0)


def test_long_segment_is_fast():
    import time
    s = " ".join(["increase the risk of conflict and reduce capacity"] * 600)  # ~4,800 words
    m = Matcher({"i": ["increas*", "rise"], "c": ["capacit*", "resilien*"]})
    t = time.perf_counter()
    _, keep = m.match([s] * 5, {"i", "c"})
    for k in range(5):
        co_occurs(s, keep["i"][k], keep["c"][k], 0)
    assert time.perf_counter() - t < 5  # v2.1 needed tens of seconds for this
