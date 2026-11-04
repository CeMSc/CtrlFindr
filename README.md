# CtrlFindr

CtrlFindr is a toolkit for dictionary-based **content analysis** of English text documents.
It splits each document into sentences and codes every sentence against an *assessment
framework* that you write in a spreadsheet (LibreOffice `.ods` or Excel `.xlsx`). The
framework combines lists of terms, proximity rules, document-level conditions, counts
and sentiment into variables, and the results are exported as tables per
document and per sentence.

Version 3.0 is a rewrite of the 2.x notebook with a new framework format; 2.x frameworks
are not read. See [CHANGELOG.md](CHANGELOG.md) and [docs/result_changes.md](docs/result_changes.md).
Version 2.1 is kept in [legacy/](legacy/).

## How it works

1. **`check`**: validates the framework, reporting every problem at once with its sheet and
   row, and runs the framework's own test sentences. No text is needed.
2. **`ingest`**: reads the `.txt` files, normalizes the text (see
   [Text preparation](#text-preparation)), splits it into sentences with
   [pySBD](https://github.com/nipunsadvilkar/pySBD), scores each sentence with
   [VADER](https://github.com/cjhutto/vaderSentiment) and stores everything in an SQLite
   database. Later runs only process new or changed files.
3. **`analyze`**: codes every sentence and writes the results, a codebook and the run
   metadata. Run it again after each framework edit; there is no need to ingest again.

`run` does ingest and analyze, `template` writes a blank framework, and `sample` draws
sentences to hand-code (see [Validation](#validation-against-hand-coding)).

## Installation

You need Python 3.11 or newer (tested with 3.11, 3.12 and 3.13), on Windows, macOS or
Linux. No GPU is needed. Download or clone this repository, then run the following in its
folder:

```
pip install -r requirements.lock
pip install -e . --no-deps
```

`requirements.lock` pins the exact package versions used to produce the reference results
and to run the tests. A virtual environment (`python -m venv .venv`) keeps these versions
apart from other projects.

## Quick start

```
python -m ctrlfindr run --config examples/example_project/ctrlfindr.toml
```

The results appear in `examples/example_project/Output/`. To start your own project:

1. Copy `examples/example_project/ctrlfindr.toml` into a new folder and put your `.txt` files
   in `Text_files/` next to it.
2. Create a framework: `python -m ctrlfindr template Input/Assessment_framework.xlsx` (or
   `.ods`), or copy one from [template/](template/). If you use `.xlsx`, change the
   `framework` path in `ctrlfindr.toml`.
3. Fill it in, using the example framework and the template's `instructions` sheet as a
   guide, and check it with `python -m ctrlfindr check --config ctrlfindr.toml`.
4. Run `python -m ctrlfindr run --config ctrlfindr.toml`.

From Jupyter, use [notebooks/CtrlFindr.ipynb](notebooks/CtrlFindr.ipynb).

**JupyterLab server.** For long runs, start the command in a JupyterLab terminal rather than
a notebook: it keeps running if the browser tab closes or an idle kernel is shut down. Keep
the database on the server's local disk (set `database` in the config); SQLite is
unreliable on network file systems, which many servers use for home folders.

## The assessment framework

A framework is one spreadsheet with these sheets (the template has them all):

| sheet | columns | purpose |
|---|---|---|
| `meta` | key, value | identifies the framework; `format_version` is required |
| `settings` | key, value, note | coding settings (optional; defaults apply) |
| `taxonomy` | list, term, note | the terms, one per row |
| `queries` | name, query, note | named expressions over lists, queries and sentiment |
| `variables` | number, name, definition, parent, note | the exported results |
| `tests` | sentence, expect_true, expect_false, group, note | example sentences checked on every run (optional) |
| `instructions` | | the rules in short; ignored by CtrlFindr |

Cells are read as displayed (`1.10` stays `1.10`; a decimal comma `1,1` is read as `1.1`).
Matching ignores upper and lower case. Notes are copied into the codebook.

### Names

Lists and queries share one set of names. A name starts with a letter and contains only
letters, digits, `_` and `-` (`fast_onset`, `co-slow`, `a1`); names are case-insensitive and
must be unique. The reserved names are:

- the operators `and`, `or`, `not`;
- the function names below;
- the sentiment flags;
- `document`, `sentences`, `sentence_id`, `start`, `end`, `vader_compound`.

Variable names are free text, because variables are referred to by number.

### meta

One key per row. `format_version` must be `1`. A framework without it, or with another
version, is rejected rather than reinterpreted. `title`, `version`, `author`, `language`
(only `en` is supported) and `description` are optional, and you may add your own keys. All
of them go into `run_metadata.json` and the codebook.

### settings

The framework holds every setting that changes how sentences are coded, so the framework
file alone defines the coding. A missing row uses the default.

| key | default | meaning |
|---|---|---|
| `sentiment` | true | score sentences with VADER and allow the sentiment flags |
| `sentiment_positive_threshold` | 0.05 | compound score at or above which a sentence is positive |
| `sentiment_negative_threshold` | -0.05 | compound score at or below which a sentence is negative |
| `aggregate_children` | false | a variable is also true wherever one of its child variables is true |
| `tests_must_pass` | true | stop the analysis when a test sentence fails |

### taxonomy: terms

One term per row (`list`, `term`, `note`). A list is true for a sentence when one of its
terms occurs in it, after exclusions.

| term | matches |
|---|---|
| `flood` | the word "flood" only, never "floods" or "floodplain": terms match whole words |
| `flood*` | `*` stands for any letters or digits, including none: "flood", "floods", "flooding". It may appear anywhere in a term (`*fire`, `biodivers* loss`) but never crosses a space or hyphen. |
| `human~wildlife` | `~` stands for a hyphen, a space or nothing: "human-wildlife", "human wildlife", "humanwildlife" |
| `climate _ risk` | `_` on its own stands for exactly one word: "climate change risk" |
| `flash flood*`, `u.s.` | multi-word terms and punctuation match literally, whatever the line breaks and spacing in the text |
| `-business climate` | an **exclusion phrase**: a match of this list that lies inside "business climate" is ignored. Other matches in the same sentence still count. |
| `-@idioms` | **excludes a list**: every term of the list `idioms` is an exclusion phrase for this list |

Exclusions work by containment. In "The business climate improved while climate change
accelerated", a list `clim` with `climat*` and `-business climate` ignores the first
"climate" and counts the second. A query `clim AND NOT idioms` would instead drop the
whole sentence.

Typographic variants are folded in both terms and text: dashes (– — ‐ → -), curly quotes,
ligatures and full-width characters. `human–wildlife` therefore also matches
"human-wildlife".

### queries: expressions

A query is a named expression, and its result is a column of the output.

- **Operators:** `AND`, `OR`, `NOT` and parentheses. `NOT` binds tighter than `AND`, and
  `AND` tighter than `OR`, so `a OR b AND c` means `a OR (b AND c)`.
- **Names:** lists, other queries (in any order; loops are reported) and sentiment flags.
  Quotes around names are optional.
- **Inline terms:** `{term}` uses a term directly, with the taxonomy syntax: `fast AND {el niño}`.

Functions:

| function | true for a sentence when… |
|---|---|
| `NEAR(a, b, n)` | a and b occur at most n words apart, in either order |
| `BEFORE(a, b, n)` | as NEAR, with a first |
| `NOT_NEAR(a, b, n)` | some match of a has no match of b within n words. `NOT NEAR(a, b, n)` instead means no pair is near. |
| `DOC(x)`, `DOC(x) >= n` | x is true in at least one (n) sentence(s) of the same document |
| `DOC_SHARE(x) >= p` | x is true in at least the share p (0–1) of the document's sentences |
| `WINDOW(x, k)` | x is true in this sentence or within k sentences before or after it, in the same document |
| `COUNT(a) >= n` | a has at least n matches in the sentence |
| `COUNT_DISTINCT(a) >= n` | at least n different terms of a match in the sentence |

Rules for the arguments:

- **Distance** is the number of words strictly between the two matches; adjacent matches
  have distance 0. Only tokens with a letter or digit count as words, so a standalone "-"
  does not. Overlapping matches never count, so a word in both lists is not near itself.
- **The `a` and `b` arguments need word positions.** They must be lists, `{terms}` or OR
  combinations of these, possibly through a query defined that way.
- **Comparisons** are `>=`, `>`, `=`, `!=`, `<=` and `<`. Without one, `DOC(x)` and
  `COUNT(x)` mean ≥ 1, and `DOC_SHARE(x)` means > 0.
- **In COUNT and COUNT_DISTINCT,** a match lying inside a longer match of the same list
  counts once ("climate change" is counted once by `climat*` and `climate change`). Each
  taxonomy row is one term for COUNT_DISTINCT.

Example queries:

```
co_slow   NEAR(slow1, slow2, 5)
a1        fast AND NOT (slow OR co_slow OR exclude_climate)
b2        a1 AND (hwc OR co_hwc) AND NOT exclude_conflict AND DOC(ipcc)
compound  COUNT_DISTINCT(hazards) >= 2 AND DOC_SHARE(climate) >= 0.05
```

### variables

| number | name | definition | parent |
|---|---|---|---|
| 1 | Climate change | a1 OR a2 | |
| 1.1 | Fast onset | a1 | |

- **number:** `1`, `1.1`, `1.1.1`, unique. In Excel, keep the column formatted as text, as
  in the template, so that `1.10` is not turned into `1.1`.
- **definition:** an expression as in queries, which may also use `V(1.1)` for another
  variable.
- **parent:** optional. When empty, `1.1` belongs to `1` and `1.1.1` to `1.1`.
- **`aggregate_children = true`:** a parent is also true wherever a child is true, and its
  definition may stay empty.
- **Hierarchy check:** with `aggregate_children = false`, the run warns when a child
  variable is true in a sentence where its parent is false.

The output columns are named `1. Climate change`, `1.1 Fast onset` and so on.

### tests

| sentence | expect_true | expect_false | group |
|---|---|---|---|
| Floods and storms destroyed the harvest. | fast, a1, 1.1 | slow, a2 | |
| The IPCC published its assessment report. | | | ipcc-doc |
| Floods led to human-wildlife conflict. | b2, 2.1.2 | b1 | ipcc-doc |

- **When the tests run:** test sentences go through the same normalization and coding as
  the corpus on every `check`, `analyze` and `run`.
- **Expectations** name lists, queries, sentiment flags or variable numbers, separated by
  commas.
- **Groups:** rows with the same `group` form one document, in order, so `DOC`, `DOC_SHARE`
  and `WINDOW` can be tested. A row in a group may carry no expectations and only
  provide context, like the IPCC sentence above.
- **Failures** name the row, the lists that matched and the values behind the failing
  query. With `tests_must_pass = true` (default) the analysis stops.

Tests protect a framework from unintended changes when you edit terms or queries.

## Validation against hand coding

Tests check that rules work as intended. Validation measures how well the framework agrees
with human judgement on real sentences:

1. `python -m ctrlfindr sample --config ctrlfindr.toml --n 50` (optionally
   `--variables 1,2.1`, `--seed 7`) writes `validation_sample.tsv` to the output folder. For
   each variable it contains 50 sentences CtrlFindr flagged and 50 it did not, mixed. It
   does not show CtrlFindr's code, so the coding stays blind.
2. Fill `human_code` with 1 or 0 and save the file as the project's validation file
   (`[paths] validation`, default `Input/validation.tsv`).
3. Every analysis then writes `validation_report.tsv` and `validation_disagreements.tsv`.

The report gives, per variable, the confusion counts, the precision and an estimated recall.
Flagged and unflagged sentences are sampled separately, so each stratum is weighted by its
size in the corpus:

- precision = TP / coded flagged;
- recall = (precision × flagged in the corpus) / (precision × flagged in the corpus + FN rate
  among coded unflagged × unflagged in the corpus);
- F1 combines the two.

This assumes random sampling within each stratum, which is what `sample` does.

## Sentiment

Each sentence gets a VADER compound score between -1 and 1, computed on the original,
not lowercased, sentence. The five flags use the thresholds from the settings sheet:

| flag | true when (default thresholds) |
|---|---|
| `sentiment_positive` | compound ≥ 0.05 |
| `sentiment_negative` | compound ≤ -0.05 |
| `sentiment_neutral` | -0.05 < compound < 0.05 |
| `sentiment_neutralpositive` | compound > -0.05 |
| `sentiment_neutralnegative` | compound < 0.05 |

A score of exactly 0 is always neutral. Points to keep in mind:

- **Mixed flags make results asymmetric.** If one query uses `sentiment_neutralpositive` and
  its counterpart uses `sentiment_negative`, neutral sentences count for the first and never
  for the second.
- **VADER was built for social-media text.** On formal prose (reports, legal texts) its
  scores are weak signals: much of the text scores as neutral, and domain words ("risk",
  "conflict", "loss") can pull scores negative whatever the stance of the sentence. Check a
  sample by hand before relying on sentiment-based variables.

## Outputs

All outputs are written to the `output` folder. In the TSV files, booleans are `True` and
`False`.

| file | content |
|---|---|
| `results_document.tsv` | per document, the number of sentences in which each variable is true |
| `results_document_clipped.tsv` | the same, capped at 1 (does the document contain the variable at all?) |
| `results_document_percentage_of_sentences.tsv` | the same as a percentage of the document's sentences |
| `results_document_sentiment.tsv` | per document, the mean compound score of the sentences in which each variable is true. **An empty cell means the variable never occurred in that document**; it is not a score of 0. Written only when sentiment is on. |
| `results_sentences.tsv` | one row per sentence: `sentence_id`, `start`, `end`, `document`, `sentences` (lowercased, normalized), every list, `vader_compound` and the sentiment flags, every query, every variable |
| `codebook.md` | the framework in readable form: meta, settings, variables, queries, lists with exclusions and notes |
| `run_metadata.json` | how the results were made: CtrlFindr and package versions, config, framework meta, settings and SHA-256, test results, SHA-256 of every text file |
| `validation_*.tsv` | see [Validation](#validation-against-hand-coding) |
| `ctrlfindr.log` | the messages of the last command-line run |

Result files that the current settings no longer produce are removed, so the folder never
mixes runs. The database (`analysis.sqlite`) has these tables:

- `documents`: path, SHA-256 and cleaned text;
- `sentences`: text, offsets and VADER score;
- `results_sentences`: a copy of the sentence results.

`clean_text[start:end]` of a document is the sentence, so a flagged sentence can be read in
context.

## Text preparation

The following is applied to every document:

- **Encoding.** Files must be UTF-8; a byte-order mark, which some Windows editors add, is
  ignored. Other encodings stop ingestion with a list of the affected files, unless
  `encoding_errors = "replace"` is set.
- **Normalization.** Unicode NFKC plus the dash and quote folding described above.
- **Line breaks.** A single line break becomes a space, so words at line ends are never
  glued together. A blank line always ends a sentence, so headings are not glued
  onto the next sentence.
- **Citation marks.** A space is inserted between a full stop and a following citation mark
  (`2050.[4] The` → `2050. [4] The`) so that the sentence splitter separates the sentences.
- **Junk segments.** Segments that contain no letter (page numbers, `[12]`, lone
  punctuation) are dropped.
- **Optional cleaning** (`text_cleaning = true`): URLs are removed, and `..` or `...`
  becomes `.`.

Segments longer than 500 words (configurable) are reported. They usually come from tables,
reference lists or text without punctuation, and they distort per-sentence percentages.

## Configuration

`ctrlfindr.toml` says where the files are and how the texts are read; how sentences are
coded is in the framework's settings sheet. See
[examples/example_project/ctrlfindr.toml](examples/example_project/ctrlfindr.toml) for
every setting with comments. Relative paths are relative to the config file's folder.

The next ingest rebuilds the database when you:
- change `text_cleaning` or `encoding_errors`;
- upgrade CtrlFindr, pySBD or vaderSentiment.

## Performance

On the development machine, the example project (7 documents, about 250,000 characters)
takes about 2 seconds for ingest and analysis together. Analysis of about 32,000 sentences
takes under a second. Sentence splitting during ingest is the slowest step, at about
200,000 characters per second, and runs once per file.

## Development

```
pip install -r requirements.lock
pip install -e . --no-deps
python -m pytest
```

The tests include the following regression checks:

- the 2.1 code in `legacy/` reproduces its committed outputs;
- the example framework, which is the 2.1 example's rules rewritten in format 1, changes
  exactly the rows explained in `docs/result_changes.md` on the 2.1 sentences;
- the full pipeline on the example project reproduces `tests/fixtures/golden_v3`.

GitHub Actions runs the tests on Windows and Linux with Python 3.11, 3.12 and 3.13.
`tools/compare_outputs.py OLD NEW` compares two output folders.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).

## How to cite

Scartozzi, Cesare M. (2023). CtrlFindr (Version 3.0). Available from https://github.com/CeMSc/CtrlFindr.
