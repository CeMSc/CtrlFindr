# Sample project: the four dimensions of food security

A small project for testing a CtrlFindr installation. The framework
(`Input/Assessment_framework.xlsx`, with an identical `.ods` copy) measures availability,
access, utilization and stability. It uses every term marker, query function and variable
feature at least once. The corpus contains the odd cases listed below. The texts are
invented for the test.

## Run it

From the repository folder:

```
python -m ctrlfindr check --config examples/food_security_project/ctrlfindr.toml
python -m ctrlfindr run --config examples/food_security_project/ctrlfindr.toml
python tools/compare_outputs.py examples/food_security_project/expected examples/food_security_project/Output
```

1. `check` should report the framework as valid and **all 29 tests passed**.
2. `run` should report: 1 hidden file skipped; `empty.txt` with no sentences; 44 sentences
   ingested; 6 sentences excluded (5 headers, 1 footnote); 38 analyzed.
3. The comparison should say **all shared cells identical** for the four document tables,
   and show the same sentence counts and totals for `results_sentences.tsv`.

Also open `Input/Assessment_framework.xlsx` in Excel. Variable numbers such as `1.1` must
stay as typed, and the dropdowns in `settings` and in the `parent` column of `variables`
should work.

## What to look for in `Output/results_sentences.tsv`

| case | file | expected |
|---|---|---|
| byte-order mark and Windows line endings | drought_report.txt | first sentence starts with "drought update", with no stray character |
| missing space after a full stop | drought_report.txt | "…by a third.The rice crop…" becomes two sentences |
| `...` and a URL | drought_report.txt | "districts." ends a sentence; the URL is gone ("see for details.") |
| citation mark after a full stop | drought_report.txt | "[4] many households…" is its own sentence |
| heading without punctuation in a .txt file | drought_report.txt, market_prices.txt | glued onto the next sentence (`.txt` is unstructured; use `.md` to keep headings apart) |
| `El Niño` (non-ASCII) | drought_report.txt | `shocks` and `stab_seasonal` are true |
| `food~aid`, `cash~transfer*` | drought_report.txt, market_prices.txt | "food-aid" and "cash transfer" match |
| term typed with an en dash (`e–voucher*`) | market_prices.txt | matches "e-voucher" |
| excluded list (`-@idioms`) | market_prices.txt | "market of ideas" does not count as a market |
| excluded phrase (`-shock absorber`) | stability_outlook.md | `shocks` is false in "shock absorber reserves are low." |
| `.TXT` extension, full stops in a term (`u.n.`) | notes.TXT | ingested; `agencies` and `precedence` are true |
| empty file | empty.txt | a warning; not in the results |
| hidden file | ._notes.txt | skipped |
| Markdown headings, list items, table rows | nutrition_brief.md, stability_outlook.md | `structure` is header (excluded), list, table or body |
| footnote and its marker `[^1]` | nutrition_brief.md | the footnote is excluded; no "1" left in the sentence |
| year at the start of a wrapped line | nutrition_brief.md | "undernutrition rates rose between 2019 and 2020." is one body sentence |
| superscript ordinal | nutrition_brief.md | "21<sup>st</sup>" reads "21st" |
| DOC, DOC_SHARE | nutrition_brief.md, stability_outlook.md | `doc_ipc` true in every sentence of both; `doc_nutrition` only in nutrition_brief |
| WINDOW | stability_outlook.md | `stab_window` true for "prices then spiked." (flood in the previous sentence) |
| NEAR vs NOT NEAR | drought_report.txt | `prices_no_shock` false for "food prices rose very sharply after the drought." (5 words apart) |
| NOT_NEAR | stability_outlook.md | `access_markets` false for "floods and conflict disrupted markets" |

The settings sheet uses a decimal comma (`0,05`) on purpose: it must be read as 0.05.
