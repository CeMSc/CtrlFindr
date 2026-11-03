# Changelog

## 3.0.0

A rewrite of the 2.1 notebook as a Python package (`ctrlfindr`) with a command line, a
thin notebook, tests and CI on Windows and Linux. The input formats (framework `.ods` and
`.txt` files) and the five output tables are kept. The effect of each change on the example
project is quantified in [docs/result_changes.md](docs/result_changes.md).

### Changes that affect results

- **Co-occurrence distance** is now the number of words *between* the two matches, as the
  framework instructions describe. 2.1 also counted the first keyword, so a distance of N
  allowed only N-1 words in between.
- **Co-occurrence self-matches** are no longer counted. In 2.1, a term present in both lists
  (for example `encroach*`) co-occurred with itself. Overlapping matches do not count either.
- **Line breaks no longer glue words together.** 2.1 deleted line breaks before capitals and
  digits (`climate security\nClimate change` → `securityclimate`).
- **Blank lines end sentences**, so headings are no longer glued onto the following sentence.
- **Sentence splitting** uses pySBD instead of NLTK punkt.
- **Segments without letters**, including citation marks such as `[12]`, are dropped. 2.1
  dropped only segments made of digits and punctuation.
- **Unicode normalization and dash/quote folding** apply to text and framework, so
  `human–wildlife` (en dash) now matches `human-wildlife` (hyphen).
- **Terms ending or starting with punctuation** (`u.s.`) now match; 2.1's `\b` boundaries
  prevented it.
- **Sentiment:**
  - scores come from vaderSentiment (the VADER authors' implementation) instead of NLTK's
    port;
  - the original-case sentence is scored, not the lowercased one;
  - positive and negative thresholds are ±0.05 instead of 0, configurable (0 reproduces
    2.x).

### Changes to inputs and outputs

- The `vadar_compound` column is renamed `vader_compound`.
- `results_sentences.tsv` starts with three new columns: `sentence_id`, `start` and `end`.
- The database file is `analysis.sqlite` (it was `database`). Source documents, sentences
  and results are separate tables. 2.1 overwrote the source table with results, so a second
  run from the database failed.
- New `run_metadata.json` with versions, settings and file hashes.
- Settings live in `ctrlfindr.toml` instead of a notebook cell. `LOAD_DATABSE` is gone:
  ingest and analysis are separate steps, and ingest only processes new or changed files.
- `.txt` extensions are matched case-insensitively, and only the extension is removed from
  document names. Files are processed in sorted order.

### Framework validation

The whole framework is checked before any text is processed, and all problems are
reported together with sheet and row:

- an empty list (2.1 flagged every sentence);
- an unknown name in a query (2.1 treated it as true);
- duplicate names across sheets (2.1 silently overwrote columns);
- blank co-occurrence names;
- non-integer distances;
- duplicate variable numbers (`1.10` is no longer read as `1.1`);
- invalid aggregation queries;
- partly filled rows;
- query loops.

Queries may now refer to other queries. Operators inside quoted names
(`"command-and-control"`) are no longer rewritten.

### Safety and robustness

- Queries are parsed and evaluated without `eval`.
- Every database connection is closed on error, and table names are fixed in code.
- Files that are not UTF-8 stop ingestion with a list of the affected files, or are
  decoded with replacement characters if configured. A UTF-8 byte-order mark is accepted.
- TSV files are written as UTF-8 with `\n` line endings on every platform.
- The framework is read without odfpy (standard library only).

### Speed

- Keyword matching uses one Aho-Corasick pass over all terms.
- Co-occurrences check only sentences where both lists matched, and only the nearest pairs.
- Queries are evaluated column-wise.

Analysis is about 40 times faster than in 2.1; on very long segments, the co-occurrence step
went from cubic to near-linear cost. `estimate_run_time` was replaced by progress bars.

### Repository

- Code: `ctrlfindr/`. Example data: `examples/example_project/`. Blank framework:
  `template/`. Reference outputs: `tests/fixtures/`.
- The 2.1 notebook is kept unchanged in `legacy/v2.1/`.
- The committed LibreOffice lock file and the generated database were removed; outputs and
  databases are now ignored by git.
