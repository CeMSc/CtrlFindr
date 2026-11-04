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
    md = "First line\nsecond line.\n\n> Quoted text.\n\n---\n\n```\ncode here\n```\n\n[^1]: Footnote text.\n[ref]: http://x.org"
    assert blocks(md) == [("body", "First line second line."), ("body", "Quoted text."), ("body", "code here"),
                          ("body", "Footnote text.")]


def test_inline_markup_is_removed():
    assert inline_text("**Bold** and *it* and _em_ but snake_case_name, `code`, ~~gone~~") == \
        "Bold and it and em but snake_case_name, code, gone"
    assert inline_text("See [the IPCC](https://ipcc.ch)[^2] ![fig](f.png) <b>now</b> &amp; \\*literal\\*") == \
        "See the IPCC now & *literal*"
