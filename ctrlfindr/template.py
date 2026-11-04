"""Blank framework templates (.ods / .xlsx) with instructions and dropdowns."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .framework import FORMAT_VERSION, SENTIMENT_TERMS, Settings
from .spreadsheet import Dropdown, SheetSpec, write_workbook

META_KEYS = ("format_version", "title", "version", "author", "language", "description")

SETTING_NOTES = {
    "sentiment": "true: score every sentence with VADER and allow the sentiment_* flags in expressions",
    "sentiment_positive_threshold": "VADER compound score at or above which a sentence is positive",
    "sentiment_negative_threshold": "VADER compound score at or below which a sentence is negative",
    "aggregate_children": "true: a variable is also true wherever one of its child variables is true",
    "tests_must_pass": "true: stop the analysis when a test in the tests sheet fails",
}

INSTRUCTIONS = [
    # (sheet or topic, rule, example)
    ("overview", "A framework tells CtrlFindr how to code every sentence. Lists of terms (taxonomy) are combined "
     "into queries, and queries into the variables that are exported. Matching ignores upper/lower case.", ""),
    ("names", "Lists and queries share one set of names: start with a letter; then letters, digits, _ or -. "
     "Each name must be unique.", "fast_onset, co-slow, a1"),
    ("names", "Reserved names: and, or, not, the function names below, the sentiment_* flags, document, "
     "sentences, sentence_id, paragraph, start, end, vader_compound.", ""),
    ("meta", "One key per row. format_version is required and must stay " + FORMAT_VERSION + ". Other keys are "
     "optional and are copied into run_metadata.json and the codebook; you may add your own.",
     "title | Climate security frameworks"),
    ("settings", "One setting per row; a missing row uses the default shown in this template.", "sentiment | true"),
] + [("settings", f"{k}: {SETTING_NOTES[k]}.", f"default: {str(v).lower()}") for k, v in asdict(Settings()).items()] + [
    ("taxonomy", "One term per row: list name, term, optional note. Terms match whole words only.", "hazards | drought"),
    ("taxonomy", "* stands for any letters or digits, also none. It can appear anywhere in a term but never "
     "crosses a space or hyphen.", "flood* -> flood, floods, flooding"),
    ("taxonomy", "~ stands for a hyphen, a space or nothing.", "human~wildlife -> human-wildlife, human wildlife"),
    ("taxonomy", "_ on its own stands for exactly one word.", "climate _ risk -> climate change risk"),
    ("taxonomy", "A term starting with - is an exclusion phrase: a match of the list inside it is ignored; other "
     "matches in the same sentence still count.", "-business climate"),
    ("taxonomy", "-@list: every term of that list is an exclusion phrase for this list.", "-@climate_idioms"),
    ("queries", "One named expression per row. Combine names with AND, OR, NOT and parentheses. NOT binds "
     "tighter than AND, AND tighter than OR. Quotes around names are optional.", "fast AND NOT (slow OR co_slow)"),
    ("queries", "Names can be lists, other queries or sentiment flags.", "a2 AND sentiment_negative"),
    ("queries", "{term} uses a term directly, with the taxonomy term syntax.", "fast AND {el niño}"),
    ("queries", "NEAR(a, b, n): a and b at most n words apart, in any order (words in between; adjacent = 0). "
     "a and b must be lists, {terms} or OR combinations of those.", "NEAR(increase, capacity, 4)"),
    ("queries", "BEFORE(a, b, n): as NEAR, with a first.", "BEFORE(drought, migration, 10)"),
    ("queries", "NOT_NEAR(a, b, n): a occurs somewhere with no b within n words.", "NOT_NEAR(conflict, wildlife, 5)"),
    ("queries", "DOC(x): x is true in at least one sentence of the same document; add >= n for at least n "
     "sentences.", "hwc AND DOC(ipcc) >= 3"),
    ("queries", "DOC_SHARE(x) >= p: x is true in at least this share (0 to 1) of the document's sentences.",
     "DOC_SHARE(climate) >= 0.05"),
    ("queries", "PARAGRAPH(x): x is true in a sentence of the same paragraph (also with >= n).", "PARAGRAPH(ipcc)"),
    ("queries", "WINDOW(x, k): x is true in this sentence or within k sentences before or after it.",
     "drought AND WINDOW(conflict, 1)"),
    ("queries", "COUNT(a) >= n: at least n matches of a in the sentence. COUNT_DISTINCT(a) >= n: at least n "
     "different terms. A match inside a longer match of the same list counts once.", "COUNT_DISTINCT(hazards) >= 2"),
    ("queries", "Comparisons: >=  >  =  !=  <=  <. Sentiment flags: " + ", ".join(SENTIMENT_TERMS) + ".", ""),
    ("variables", "The exported results. number: 1, 1.1, 1.1.1 (unique; type them as text). name: free text.", "1.1 | fast onset"),
    ("variables", "definition: an expression as in queries; V(n) uses another variable.", "a1 OR V(1.2)"),
    ("variables", "parent: optional. When empty, 1.1 belongs to 1 and 1.1.1 to 1.1.", ""),
    ("variables", "With aggregate_children = true a parent is also true wherever a child is true, and its definition "
     "may stay empty.", ""),
    ("tests", "Example sentences checked on every run. expect_true / expect_false: lists, queries, sentiment flags "
     "or variable numbers, separated by commas.", "Floods hit the coast. | fast, 1.1 | slow"),
    ("tests", "group: rows with the same group form one document, in order (for DOC, WINDOW, PARAGRAPH). "
     "paragraph: optional paragraph number within the group.", "group: doc1"),
    ("instructions", "This sheet is ignored by CtrlFindr.", ""),
]


def template_sheets() -> list[SheetSpec]:
    settings_rows = [["key", "value", "note"]] + [
        [k, str(v).lower(), SETTING_NOTES[k]] for k, v in asdict(Settings()).items()
    ]
    return [
        SheetSpec("instructions", [["sheet", "rule", "example"]] + [list(r) for r in INSTRUCTIONS], [3, 17, 8], wrap=True),
        SheetSpec("meta", [["key", "value"], ["format_version", FORMAT_VERSION], ["title", ""], ["version", ""],
                           ["author", ""], ["language", "en"], ["description", ""]], [5, 14],
                  dropdowns=[Dropdown(0, META_KEYS, rows=40)]),
        SheetSpec("settings", settings_rows, [7, 3, 16],
                  dropdowns=[Dropdown(0, tuple(asdict(Settings())), rows=20), Dropdown(1, ("true", "false"), rows=20)]),
        SheetSpec("taxonomy", [["list", "term", "note"]], [5, 7, 12]),
        SheetSpec("queries", [["name", "query", "note"]], [5, 16, 10]),
        SheetSpec("variables", [["number", "name", "definition", "parent", "note"]], [2.5, 8, 12, 2.5, 8],
                  dropdowns=[Dropdown(3, source=("variables", "A"), rows=500)]),
        SheetSpec("tests", [["sentence", "expect_true", "expect_false", "group", "paragraph", "note"]],
                  [14, 6, 6, 3, 2.5, 8]),
    ]


def write_template(path: str | Path) -> Path:
    """Write a blank framework (.ods or .xlsx)."""
    return write_workbook(path, template_sheets())
