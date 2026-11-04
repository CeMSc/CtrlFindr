import re
import time

import pytest

from ctrlfindr.matching import (ListSpec, Matcher, TermError, count, count_distinct, near, not_near, parse_term,
                                word_tokens)


def found(lists, sentence):
    m = Matcher({k: v if isinstance(v, ListSpec) else ListSpec(v) for k, v in lists.items()})
    return {name: [sentence[s:e] for s, e, _ in ms] for name, ms in zip(m.lists, m.matches(sentence))}


@pytest.mark.parametrize("term,sentence,hit", [
    ("flood", "the flood came", True), ("flood", "the floods came", False), ("flood*", "the floods came", True),
    ("flood*", "flash-flooding", True), ("flash flood*", "a flash flooding event", True),
    ("climat*", "anticlimatic", False), ("u.s.", "the u.s. policy", True), ("biodivers* loss", "biodiversity loss", True),
    ("*fire", "a bushfire", True), ("co2", "co2 emissions", True), ("niño", "el niño year", True),
    ("niño", "el niños year", False), ("ipcc", "the ipcc_report", False),
    ("human~wildlife", "human-wildlife conflict", True), ("human~wildlife", "human wildlife conflict", True),
    ("human~wildlife", "humanwildlife conflict", True), ("human~wildlife", "human/wildlife", False),
    ("climate _ risk", "climate change risk", True), ("climate _ risk", "climate risk", False),
    ("climate _ risk", "climate and other risk", False), ("adapt* _ _ change", "adapting to climate change", True),
])
def test_term_semantics(term, sentence, hit):
    assert bool(found({"l": [term]}, sentence)["l"]) is hit
    assert bool(parse_term(term).regex.search(sentence)) is hit  # automaton == regex definition


def test_automaton_equals_regex_on_mixed_terms():
    terms = ["climate change", "climat*", "warm*", "heat", "heat wave*", "sea level*", "el niño", "human~wildlife",
             "climate _ risk", "*fire"]
    rx = re.compile("|".join(parse_term(t).regex.pattern for t in terms))
    for s in ["the climate is changing", "heatwaves and heat waves", "sea levels", "el niño and la niña", "nothing",
              "climatechange", "human wildlife", "climate change risk", "wildfire", "heat-wave"]:
        assert bool(found({"l": terms}, s)["l"]) == bool(rx.search(s)), s


@pytest.mark.parametrize("bad", ["*", "_", "_ risk", "climate _", "~x", "x~", "a ~ b", "a~~b"])
def test_invalid_terms(bad):
    with pytest.raises(TermError):
        parse_term(bad)


def test_exclusion_phrase_and_list_by_containment():
    lists = {"clim": ListSpec(["climat*"], ["business climate"], ["idioms"]), "idioms": ListSpec(["climate of fear"])}
    s = "the business climate and the climate of fear grew while climate change worsened"
    assert found(lists, s)["clim"] == ["climate"]  # only the last mention counts
    assert found(lists, "the business climate improved")["clim"] == []


def test_exclusion_only_removes_contained_matches():
    lists = {"clim": ListSpec(["climate change"], ["climate"])}
    assert found(lists, "climate change")["clim"] == ["climate change"]  # longer than the exclusion


def test_exclusion_loop_is_rejected():
    with pytest.raises(TermError, match="loop"):
        Matcher({"a": ListSpec(["x"], [], ["b"]), "b": ListSpec(["y"], [], ["a"])})


def test_entry_ids_are_unique_across_lists():
    m = Matcher({"a": ListSpec(["flood*"]), "b": ListSpec(["drought*"])})
    fa, fb = m.matches("floods and droughts")
    assert count_distinct(fa + fb) == 2


def test_count_and_count_distinct():
    m = Matcher({"h": ListSpec(["flood*", "drought*", "climat*", "climate change"])})
    (h,) = m.matches("floods, more floods and drought")
    assert count(h) == 3 and count_distinct(h) == 2
    (c,) = m.matches("climate change")  # climat* inside climate change counts once
    assert count(c) == 1 and count_distinct(c) == 1


def span(s, word, entry=0):
    i = s.index(word)
    return (i, i + len(word), entry)


def test_near_distance_is_words_between():
    s = "increase the adaptive capacity"
    t = word_tokens(s)
    a, b = [span(s, "increase")], [span(s, "capacity")]
    assert near(t, a, b, 2) and not near(t, a, b, 1)
    assert near(t, b, a, 2)  # any order
    assert not near(t, b, a, 2, ordered=True) and near(t, a, b, 2, ordered=True)


def test_near_ignores_punctuation_and_overlap():
    s = "capacity - which we increase"
    t = word_tokens(s)
    assert near(t, [span(s, "increase")], [span(s, "capacity")], 2)
    same = [span(s, "capacity")]
    assert not near(t, same, same, 5)  # a word never co-occurs with itself


def test_not_near():
    s = "conflict here, then many many many many words, then conflict near wildlife"
    t = word_tokens(s)
    first, second = (s.index("conflict"), s.index("conflict") + 8, 0), (s.rindex("conflict"), s.rindex("conflict") + 8, 0)
    w = [span(s, "wildlife")]
    assert not_near(t, [first, second], w, 2)      # the first conflict has no wildlife nearby
    assert not not_near(t, [second], w, 2)
    assert not_near(t, [second], [], 2) and not not_near(t, [], w, 2)


def test_word_tokens_linear_on_punctuation_runs():
    s = "increase - the, (capacity) " + "=" * 200_000 + " end"
    t0 = time.perf_counter()
    starts, ends = word_tokens(s)
    assert time.perf_counter() - t0 < 1
    assert [s[a:b] for a, b in zip(starts, ends)] == ["increase", "the,", "(capacity)", "end"]
