from ctrlfindr.markdown import blocks, inline_text


def test_headings_atx_and_setext():
    md = "# Title #\n\n## Sub\n\nSetext one\n==========\n\nSetext two\n----------\n\nBody."
    assert blocks(md) == [("header", "Title"), ("header", "Sub"), ("header", "Setext one"),
                          ("header", "Setext two"), ("body", "Body.")]


def test_lists_with_continuation_and_numbers():
    md = "- one\n  continued\n* two\n+ three\n1. four\n2) five\n• six"
    assert blocks(md) == [("list", "one continued"), ("list", "two"), ("list", "three"),
                          ("list", "four"), ("list", "five"), ("list", "six")]


def test_tables_drop_the_separator_row():
    md = "| A | B |\n|:--|--:|\n| 1 | x |\n| 2 | y |\n\nAfter."
    assert blocks(md) == [("table", "A | B"), ("table", "1 | x"), ("table", "2 | y"), ("body", "After.")]


def test_a_pipe_without_separator_row_is_not_a_table():
    assert blocks("This | that.") == [("body", "This | that.")]


def test_body_paragraphs_quotes_code_and_rules():
    md = "First line\nsecond line.\n\n> Quoted text.\n\n---\n\n```\ncode here\n```\n\n[ref]: http://x.org"
    assert blocks(md) == [("body", "First line second line."), ("body", "Quoted text."), ("body", "code here")]


def test_footnotes_are_labelled_with_their_continuation():
    md = ("Text.[^1]\n\n[^1]: First line\n    continued.\n\n    Second paragraph.\n[^2]: Other note.\n- item\n\n"
          "Body again.\n\n    indented code is body, not a footnote")
    assert blocks(md) == [("body", "Text."), ("footnote", "First line continued."), ("footnote", "Second paragraph."),
                          ("footnote", "Other note."), ("list", "item"), ("body", "Body again."),
                          ("body", "indented code is body, not a footnote")]


def test_footnote_markers_are_removed():
    assert inline_text("Emissions fell.[^1] The IPCC agreed.<sup>12</sup> Others disagreed.^3^ <SUP>a, b</SUP>") == \
        "Emissions fell. The IPCC agreed. Others disagreed."


def test_inline_markup_is_removed():
    assert inline_text("**Bold** and *it* and _em_ but snake_case_name, `code`, ~~gone~~") == \
        "Bold and it and em but snake_case_name, code, gone"
    assert inline_text("See [the IPCC](https://ipcc.ch)[^2] ![fig](f.png) <b>now</b> &amp; \\*literal\\*") == \
        "See the IPCC now & *literal*"


def test_a_year_at_the_start_of_a_wrapped_line_stays_in_the_paragraph():
    assert blocks("Temperatures rose between 2015 and\n2020. Floods followed.") == [
        ("body", "Temperatures rose between 2015 and 2020. Floods followed.")]
    assert blocks("Steps:\n1. Plan\n2. Act") == [("body", "Steps:"), ("list", "Plan"), ("list", "Act")]


def test_superscripts():
    assert inline_text("the 20<sup>th</sup> century and the 21^st^") == "the 20th century and the 21st"
    assert inline_text("10<sup>6</sup> people, 10^9^ tonnes") == "10^6 people, 10^9 tonnes"
    assert inline_text("agreed.<sup>12</sup> Notes<sup>a, b</sup> and a <sup>Registered</sup> mark") == \
        "agreed. Notes and a Registered mark"
