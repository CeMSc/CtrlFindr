# Changelog

## 3.0.0

A rewrite of the 2.1 notebook as a Python package (`ctrlfindr`) with a command line, a thin
notebook, tests and CI on Windows and Linux, and a **new framework format**. 2.x frameworks
are not read; the 2.1 example was rewritten in the new format with the same rules. The
effect of each change on the example project is quantified in
[docs/result_changes.md](docs/result_changes.md).

### New assessment framework (format 1)

- **Sheets:** `meta`, `settings`, `taxonomy`, `queries`, `variables`, `tests`, `instructions`.
  The `co_occurrences`, `doc_conditionals` and `search_strings` sheets are gone: proximity
  and document conditions are functions inside queries.
- **File types:** `.ods` or `.xlsx`. Cells are read as displayed, and `ctrlfindr template`
  writes a blank framework with instructions and dropdowns.
- **meta:** a required `format_version`, so a framework is never silently reinterpreted.
  Optional `title`, `version`, `author`, `language` and `description`, plus any key of your
  own, all copied into the codebook and `run_metadata.json`.
- **settings:** sentiment on/off and thresholds, `aggregate_children` and `tests_must_pass`.
  These live in the framework, so the framework alone defines the coding; the config only
  says where files are and how texts are read.
- **taxonomy:** one row per term (`list | term | note`). New term syntax:
  - `~` for a hyphen, space or nothing (`human~wildlife`);
  - a standalone `_` for one word (`climate _ risk`);
  - exclusion phrases (`-business climate`) and excluded lists (`-@idioms`), which drop only
    the matches that lie inside them, not the whole sentence.
- **queries:**
  - names with or without quotes;
  - `{term}` for inline terms;
  - `NEAR`, `BEFORE`, `NOT_NEAR`;
  - `DOC`, `DOC_SHARE`, `WINDOW`;
  - `COUNT` and `COUNT_DISTINCT` with comparisons;
  - queries may use other queries.
- **variables:**
  - full expressions instead of comma lists, and `V(n)` for other variables;
  - an optional `parent` column (otherwise the hierarchy follows the numbering), with
    optional aggregation of children;
  - a warning when a child is true where its parent is false.
- **tests:** example sentences with expected results, checked on every run. Groups form
  documents, so document-level rules can be tested too.
- **Validation against hand coding:** `ctrlfindr sample` writes a blind, stratified sample
  to code. Every analysis then reports precision, estimated recall and F1 per variable,
  with the disagreements.
- **Names** follow one rule (a letter first, then letters, digits, `_`, `-`), with suggested
  fixes. All problems are reported together with sheet and row, before any text is
  processed.
- **New output:** `codebook.md` with the framework in readable form.

### Text inputs and sentiment

- **Two input types:** `.txt` files are read as unstructured text, and `.md` files as
  Markdown.
- **Structure labels:** every sentence of a Markdown file is labelled `header`, `body`, `list`,
  `table` or `footnote`, and every sentence of a text file `unstructured`. Footnote markers
  (`[^1]`, `<sup>12</sup>`, `^12^`) are removed from sentences. The label is stored and
  exported in the `structure` column.
- **Excluding sentence types:** the setting `exclude_sentence_types` leaves sentence types
  out of the analysis.
- **VADER is optional:** sentiment is off unless a framework sets `sentiment = true`;
  vaderSentiment is an optional package; scores are computed during the analysis and cached
  in the database.

### Changes that affect results

- **Co-occurrence distance** (now `NEAR`) is the number of words *between* the two matches.
  2.1 also counted the first keyword, so a distance of N allowed only N-1 words in between.
- **Self-matches are no longer counted.** In 2.1, a term present in both lists (for example
  `encroach*`) co-occurred with itself. Overlapping matches do not count either.
- **Line breaks no longer glue words together.** 2.1 deleted line breaks before capitals and
  digits (`climate security\nClimate change` → `securityclimate`).
- **Blank lines end sentences**, so headings are no longer glued onto the following sentence.
- **Sentence splitting** uses pySBD instead of NLTK punkt.
- **Segments without letters**, including citation marks such as `[12]`, are dropped.
- **Unicode normalization and dash/quote folding** apply to text and framework, so
  `human–wildlife` (en dash) matches `human-wildlife` (hyphen).
- **Terms ending or starting with punctuation** (`u.s.`) match; 2.1's `\b` boundaries
  prevented it.
- **Sentiment:**
  - scores come from vaderSentiment (the VADER authors' implementation) instead of NLTK's
    port;
  - the original-case sentence is scored, not the lowercased one;
  - the thresholds are ±0.05 instead of 0, and can be changed in the framework's settings.

### Running large corpora

- **Ingest saves each file as soon as it is done.** A stopped or crashed ingest keeps every
  finished file, and the next run continues with the rest.
- **The analysis runs in batches of whole documents** (`[analysis] batch_sentences`), so
  memory use stays flat. Outputs are replaced only when a run completes, and sentiment
  scores are kept across stops.
- **`.txt` cleaning repairs a missing space after a full stop** ("published.The minister").

### Changes to inputs and outputs

- `vadar_compound` is renamed `vader_compound`.
- `results_sentences.tsv` gains `sentence_id`, `structure`, `start` and `end`.
- Variable names keep their capitalization in output headers.
- The database file is `analysis.sqlite`. Source documents, sentences and results are
  separate tables. 2.1 overwrote the source table with results, so a second run failed.
- New `run_metadata.json` with versions, settings, test results and file hashes.
- The project configuration is `ctrlfindr.toml`. Ingest and analysis are separate steps, and
  ingest only processes new or changed files.
- `.txt` extensions are matched case-insensitively, only the extension is removed from
  document names, and files are processed in sorted order.
- Result files that the current settings no longer produce are removed from the output folder.

### Safety and robustness

- Expressions are parsed and evaluated without `eval`.
- Every database connection is closed on error, and table names are fixed in code.
- Files that are not UTF-8 stop ingestion with a list of the affected files, or are decoded
  with replacement characters if configured. A UTF-8 byte-order mark is accepted.
- TSV files are written as UTF-8 with `\n` line endings on every platform.
- Ingest hashes files without loading the whole corpus into memory.

### Speed

- Keyword matching uses one Aho-Corasick pass over all terms.
- Proximity checks only sentences where both sides matched, and only the nearest pairs.
- Expressions are evaluated column-wise.

Analysis is about 40 times faster than in 2.1. On very long segments, proximity went from
cubic to near-linear cost.

### Repository

- Code: `ctrlfindr/`. Example: `examples/example_project/`. Blank frameworks: `template/`.
  Reference outputs: `tests/fixtures/`.
- The 2.1 notebook is kept unchanged in `legacy/v2.1/`. Its example framework is at
  `tests/fixtures/golden_v2.1/Assessment_framework_v2.1.ods`.
- Outputs and databases are ignored by git.
