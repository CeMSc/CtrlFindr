# CtrlFindr

CtrlFindr is a toolkit for dictionary-based **content analysis** of English text documents.
It splits each document into sentences and codes every sentence against an assessment
framework that you write in a spreadsheet. The framework holds keyword lists,
co-occurrences, document-level conditions, sentiment and Boolean queries. The results are
exported as tables per document and per sentence.

Version 3.0 is a rewrite of the 2.x notebook. The inputs (framework spreadsheet and `.txt`
files) and the five output tables are the same. Some results change because bugs were fixed
and libraries were replaced: see [CHANGELOG.md](CHANGELOG.md) and
[docs/result_changes.md](docs/result_changes.md). Version 2.1 is kept in [legacy/](legacy/).

## How it works

1. **Ingest** (`python -m ctrlfindr ingest`):
   - reads the `.txt` files and normalizes the text (see [Text preparation](#text-preparation));
   - splits it into sentences with [pySBD](https://github.com/nipunsadvilkar/pySBD);
   - scores each sentence with [VADER](https://github.com/cjhutto/vaderSentiment);
   - stores everything in an SQLite database.

   Only new or changed files are processed on later runs.
2. **Analyze** (`python -m ctrlfindr analyze`):
   - checks the whole framework and reports every problem at once;
   - codes all sentences;
   - writes the results.

   Run it again after each framework edit; there is no need to ingest again.

`python -m ctrlfindr run` does both.

## Installation

You need Python 3.11 or newer, on Windows, macOS or Linux. No GPU is needed. Download or
clone this repository, then run the following in its folder:

```
pip install -r requirements.lock
pip install -e . --no-deps
```

`requirements.lock` pins the exact package versions used to produce the reference results
and to run the tests. It is recommended for work you want to reproduce. Using a virtual
environment (`python -m venv .venv`) keeps these versions separate from other projects.

## Quick start

```
python -m ctrlfindr run --config examples/example_project/ctrlfindr.toml
```

The results appear in `examples/example_project/Output/`. To start your own project:

1. Copy `examples/example_project/ctrlfindr.toml` into a new folder.
2. Put your `.txt` files in `Text_files/`.
3. Copy `template/Assessment_framework.ods` to `Input/` and fill it in.
4. Run `python -m ctrlfindr run --config path/to/ctrlfindr.toml`.

From Jupyter, use [notebooks/CtrlFindr.ipynb](notebooks/CtrlFindr.ipynb), which calls the
same two steps.

**JupyterLab server.** For long runs, start the command above in a JupyterLab terminal
rather than a notebook: it keeps running if the browser tab closes or the kernel is shut
down for being idle. Keep the database on the server's local disk (set `database` in the
config). SQLite is unreliable on network file systems, which many servers use for home
folders.

## The assessment framework

The framework is an OpenDocument spreadsheet (`.ods`, made with LibreOffice; Excel can
save in this format) with five sheets. Keep the column headers of the template.
Everything is case-insensitive. Cells are read as displayed, so `1.10` stays `1.10`.

### taxonomy

Each column is a **list**: the header is the list name and the cells below are its terms.
A list is true for a sentence if any of its terms occurs in it.

- **Whole words only.** `flood` matches "flood" but not "floods" or "floodplain".
- **`*` matches any run of letters or digits, including none.** `flood*` matches "flood",
  "floods" and "flooding". It does not cross a space or hyphen: `flood*` finds the "flood"
  in "flash-flood", and `flash flood*` matches "flash floods". `*` may appear anywhere in
  a term (`biodivers* loss`, `*fire`).
- **Multi-word terms** match regardless of line breaks or extra spaces in the text.
- **Typographic variants are folded** in both terms and text: dashes (– — ‐ → -), curly
  quotes, ligatures, full-width characters. `human–wildlife` therefore matches
  "human-wildlife".
- **Punctuation inside or at the edge of a term** is matched literally (`u.s.`, `co2`).
- A list without terms is an error, because it would otherwise match nothing or everything.

### co_occurrences

| name of co-occurrence | first list | distance between lists | second list |
|---|---|---|---|
| co_increasecapacity | increase | 4 | capacity |

A co-occurrence is true when a term of the first list and a term of the second list occur
in the same sentence, in either order, with **at most N words between them**. Adjacent
terms have distance 0. Only tokens with a letter or digit count as words, so a standalone
"-" does not. Two matches that overlap (for example the same word listed in both lists) do
not count. The distance must be a whole number of 0 or more.

### doc_conditionals

A document-level conditional is true for **every sentence of a document** in which its list
occurs at least once. Use it for queries such as "sentences about X in documents that
mention the IPCC".

### search_strings

| query name | query |
|---|---|
| a1 | ("fast" AND NOT("slow" OR "co_slow" OR "exclude_climate")) |

A query combines names in double quotes with `AND`, `OR`, `NOT` and parentheses.

- **Names** can be lists, co-occurrences, document-level conditionals, sentiment flags
  (below) or **other queries**. Referring to another query lets you reuse it; loops are
  reported as errors.
- **Precedence:** `NOT` binds tighter than `AND`, and `AND` tighter than `OR`.
  `"a" OR "b" AND "c"` means `"a" OR ("b" AND "c")`. Use parentheses when in doubt.
- **Unknown names are errors.** Every name must exist.
- **Curly quotes**, which LibreOffice inserts automatically, are accepted.

### variables

| variable | variable number | aggregation query |
|---|---|---|
| climate change | 1 | "a1", "a2" |

A variable is true for a sentence if **any** of the listed queries is true. List query
names in double quotes, separated by commas; other operators are not allowed here. The
output column is the number followed by the name (`1. climate change`, `1.1 fast climate
onset`). Variable numbers must be unique.

**Names must be unique across all sheets**, because every list, co-occurrence, conditional,
query and variable becomes a column of the same table. These names are reserved:
`document`, `sentences`, `sentence_id`, `start`, `end`, `vader_compound` and the five
`sentiment_*` flags.

## Sentiment

Each sentence gets a VADER compound score between -1 and 1, computed on the original
(not lowercased) sentence. Five flags are available in queries:

| flag | true when (default thresholds) |
|---|---|
| `sentiment_positive` | compound ≥ 0.05 |
| `sentiment_negative` | compound ≤ -0.05 |
| `sentiment_neutral` | -0.05 < compound < 0.05 |
| `sentiment_neutralpositive` | compound > -0.05 |
| `sentiment_neutralnegative` | compound < 0.05 |

The ±0.05 thresholds are the convention recommended by VADER's authors, and you can change
them in the config. With thresholds of 0 you get the CtrlFindr 2.x behaviour: positive > 0,
neutral = 0, negative < 0.

Points to keep in mind:

- **Mixed flags make results asymmetric.** In the example framework, query d1 ("increase of
  adaptive capacity") requires `sentiment_neutralpositive` while d2 ("decrease") requires
  `sentiment_negative`. Neutral sentences therefore count toward d1 but never toward d2.
- **VADER was built for social-media text.** On formal prose (reports, legal texts) its
  scores are weak signals: much of the text scores as neutral, and the domain vocabulary
  ("risk", "conflict", "loss") can pull scores negative whatever the stance of the sentence.
  Validate a sample by hand before relying on sentiment-based variables.

## Outputs

All outputs are written to the `output` folder. In the TSV files, booleans are `True` and
`False`.

| file | content |
|---|---|
| `results_document.tsv` | per document, the number of sentences in which each variable is true |
| `results_document_clipped.tsv` | the same, capped at 1 (does the document contain the variable at all?) |
| `results_document_percentage_of_sentences.tsv` | the same as a percentage of the document's sentences |
| `results_document_sentiment.tsv` | per document, the mean compound score of the sentences in which each variable is true. **An empty cell means the variable never occurred in that document** (there is no score to average); it is not a score of 0. |
| `results_sentences.tsv` | one row per sentence. `sentence_id`, `start` and `end` locate the sentence (see below); `sentences` is the lowercased, normalized sentence; then every list, co-occurrence, conditional, `vader_compound`, sentiment flag, query and variable |
| `run_metadata.json` | when and how the results were made: CtrlFindr and package versions, settings, SHA-256 of the framework and of every text file |
| `ctrlfindr.log` | the messages of the last command-line run |

The database (`analysis.sqlite` by default) contains these tables:

- `documents`: file path, SHA-256 and the cleaned text;
- `sentences`: sentence text, offsets and VADER score;
- `results_sentences`: a copy of the sentence results.

To read a flagged sentence in context, take `clean_text[start:end]` of its document. Any
SQLite viewer, or `pandas.read_sql`, can open the database.

## Text preparation

The following is applied to every document:

- **Encoding.** Files must be UTF-8; a byte-order mark, which some Windows editors add, is
  ignored. Other encodings stop ingestion with a list of the affected files, unless
  `encoding_errors = "replace"` is set.
- **Normalization.** Unicode NFKC plus the dash and quote folding described above.
- **Line breaks.** A single line break becomes a space, so words at line ends are never
  glued together. A blank line ends a paragraph, and sentences never cross paragraphs.
- **Citation marks.** A space is inserted between a full stop and a following citation mark
  (`2050.[4] The` → `2050. [4] The`) so that the sentence splitter separates the sentences.
- **Junk segments.** Segments that contain no letter (page numbers, `[12]`, lone
  punctuation) are dropped.
- **Optional cleaning** (`text_cleaning = true`): URLs are removed, and `..` or `...`
  becomes `.`.

Segments longer than 500 words (configurable) are reported. They usually come from tables,
reference lists or text without punctuation. They distort per-sentence percentages and are
worth checking.

## Configuration

See [examples/example_project/ctrlfindr.toml](examples/example_project/ctrlfindr.toml) for
every setting with comments. Relative paths are relative to the config file's folder. When
you change `text_cleaning` or `encoding_errors`, or upgrade CtrlFindr, pySBD or
vaderSentiment, the next ingest rebuilds the database.

## Performance

On the development machine, the example corpus (7 documents, about 250,000 characters) takes about 1.5
seconds for ingest and analysis together. Analysis of about 32,000 sentences takes under a
second, against about 19 seconds for version 2.1. Sentence splitting during ingest is the
slowest step, at about 200,000 characters per second. It runs once per file.

## Development

```
pip install -r requirements.lock
pip install -e . --no-deps
python -m pytest
```

The tests include two regression checks:

- the analysis step on the sentences committed with 2.1 must change exactly the rows
  explained in `docs/result_changes.md`;
- the full pipeline on the example project must reproduce `tests/fixtures/golden_v3`.

To compare two output folders, run `tools/compare_outputs.py OLD NEW`. GitHub Actions runs
the tests on Windows and Linux with Python 3.11 and 3.13.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).

## How to cite

Scartozzi, Cesare M. (2023). CtrlFindr (Version 3.0). Available from https://github.com/CeMSc/CtrlFindr.
