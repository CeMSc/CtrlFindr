# How results changed from 2.1 to 3.0

This report covers the example project (`examples/example_project`). It separates the
causes, so that each difference can be traced to one change. Every difference listed here
is intended; none of them is suppressed in the tests.

## A. Same sentences, new analysis

Here the 3.0 analysis step was run on the sentences committed with 2.1, with the 2.1
sentiment scores and the 2.1 thresholds (0). Sentence splitting and sentiment scoring are
therefore identical, and only the matching and co-occurrence rules differ.
`tests/test_golden.py` checks that exactly these rows change
(`tests/fixtures/v21_sentences_expected_changes.json`).

| column | sentences changed | cause |
|---|---:|---|
| `hwc` | 21 | dash folding: taxonomy term "human–wildlife" (en dash) now matches "human-wildlife" (hyphen) |
| `co_slow` | 8 | distance = words strictly between the two matches (v2.1 also counted the first keyword) |
| `co_hwc` | 1 | distance = words strictly between the two matches (v2.1 also counted the first keyword) |
| `co_resourceandcompetition` | 2 | distance = words strictly between the two matches (v2.1 also counted the first keyword) |
| `co_increasecapacity` | 1 | distance = words strictly between the two matches (v2.1 also counted the first keyword) |
| `a1` | 3 | follows from the changes above |
| `a2` | 3 | follows from the changes above |
| `b2` | 1 | follows from the changes above |
| `b3` | 1 | follows from the changes above |
| `c1` | 1 | follows from the changes above |
| `d1` | 1 | follows from the changes above |
| `1.1 fast climate onset` | 3 | follows from the changes above |
| `1.2 slow climate onset` | 3 | follows from the changes above |
| `2.1 human-wildlife conflict and fast climate onset` | 1 | follows from the changes above |
| `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` | 1 | follows from the changes above |
| `2.2 human-wildlife conflict and slow climate onsets` | 1 | follows from the changes above |
| `3. resource competition and human-wild life conflict` | 1 | follows from the changes above |
| `4. increase of adaptive capacity` | 1 | follows from the changes above |

Notes:

- **hwc (+21):** the taxonomy spells the term with an en dash (`human–wildlife`), while the
  text uses a hyphen (`human-wildlife`). In 2.1 the term never matched. All 21 sentences
  contain `human-wildlife`.
- **co_\* (+12):** each newly flagged pair is exactly N words apart. 2.1 counted the first
  keyword as one of the N words. 3.0 counts only the words between the two matches, which is
  what the `How_to_fill` sheet describes. A rule with distance 5 now allows 5 words in between,
  not 4.
- **Self-matches:** dropping a co-occurrence of a word with itself (for example `encroach*`,
  which is in both `hwc1` and `compete2`) changes no sentence in this corpus. The rule is
  covered by unit tests.
- **Queries and variables:** these change only through the columns above. For example, a1
  drops by 3 because it excludes `co_slow`, which gained 8.
- **Sentence text:** the `sentences` column differs only by normalization (dashes, quotes,
  NFKC) in 101 sentences.
- **Document tables:** all other cells of the four document tables are identical.

## B. Full pipeline

### Sentence splitting and cleaning

The example corpus goes from 1,589 to 1,692 sentences (+6.5%), for these reasons:

- **Line breaks.** 2.1 deleted line breaks that were followed by a capital letter or digit,
  which glued words together (`securityclimate`, `historyhuman`, `definitionclimate`, ...).
  Such words never match. 3.0 turns every line break into a space.
- **Paragraphs.** A blank line now always ends a sentence. Headings such as "See also" or
  "Terminology" become their own short segments; in 2.1 they were glued onto the next
  sentence.
- **Splitter.** pySBD replaces NLTK punkt. Its boundaries differ in places (abbreviations,
  decimals). pySBD does not split `2050.[4] The`, so 3.0 inserts a space between a full stop
  and a following citation mark before splitting.
- **Junk segments.** Segments without any letter are dropped. This keeps 2.1's rule (numbers
  or punctuation only) and adds citation marks such as `[12]` or `[3][4]`, which pySBD splits
  off as separate segments.

Because documents have more sentences, the percentage-of-sentences values fall even where
counts rise.

### Sentiment

| change | effect on the example corpus |
|---|---|
| vaderSentiment (the VADER authors' reference implementation) instead of NLTK's port | 57 of 1,589 scores differ on the same lowercased sentences; 13 change sign. The differences are mostly in sentences with "but" or negations, which the two implementations weight differently. |
| VADER scores the original-case sentence instead of the lowercased one | 3 of 1,692 scores differ (VADER boosts words in capitals) |
| thresholds ±0.05 instead of 0 | on the 3.0 scores: positive 687 → 678, negative 494 → 483, neutral 511 → 531 |

The `vadar_compound` column is renamed `vader_compound`.

### New columns

`results_sentences.tsv` starts with `sentence_id`, `start` and `end`. With the document
name, they locate each sentence in `documents.clean_text` in `analysis.sqlite`
(`clean_text[start:end]`).

### Comparison of the committed outputs

Generated with `python tools/compare_outputs.py tests/fixtures/golden_v2.1 tests/fixtures/golden_v3 --rename vadar_compound=vader_compound`.

#### results_document.tsv
- `1. climate change` (5 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 84 → 86; wikiAnthropocene: 51 → 53; wikiClimate_change: 324 → 338; wikiClimate_justice: 115 → 120; wikiClimate_security: 161 → 165
- `1.1 fast climate onset` (3 of 7 documents): examplefile: 1 → 0; wikiClimate_change: 32 → 31; wikiClimate_justice: 6 → 5
- `1.2 slow climate onset` (6 of 7 documents): examplefile: 4 → 5; wiki2023_United_Nations_Climate_Change_Conference: 83 → 85; wikiAnthropocene: 51 → 53; wikiClimate_change: 292 → 307; wikiClimate_justice: 109 → 115; wikiClimate_security: 158 → 162
- `2.1 human-wildlife conflict and fast climate onset` (1 of 7 documents): examplefile: 1 → 0
- `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` (1 of 7 documents): examplefile: 1 → 0
- `2.2 human-wildlife conflict and slow climate onsets` (1 of 7 documents): examplefile: 1 → 2
- `3. resource competition and human-wild life conflict` (1 of 7 documents): wikiHuman-wildlife_conflict: 3 → 4
- `4. increase of adaptive capacity` (1 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 0 → 1

#### results_document_clipped.tsv
- `1.1 fast climate onset` (1 of 7 documents): examplefile: 1 → 0
- `2.1 human-wildlife conflict and fast climate onset` (1 of 7 documents): examplefile: 1 → 0
- `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` (1 of 7 documents): examplefile: 1 → 0
- `4. increase of adaptive capacity` (1 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 0 → 1

#### results_document_percentage_of_sentences.tsv
- `1. climate change` (6 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 37.33 → 35.98; wikiAnthropocene: 20.82 → 20.78; wikiClimate_change: 57.24 → 56.71; wikiClimate_justice: 60.21 → 57.14; wikiClimate_security: 70 → 66.27; wikiHuman-wildlife_conflict: 1.667 → 1.527
- `1.1 fast climate onset` (6 of 7 documents): examplefile: 8.333 → 0; wiki2023_United_Nations_Climate_Change_Conference: 0.4444 → 0.4184; wikiClimate_change: 5.654 → 5.201; wikiClimate_justice: 3.141 → 2.381; wikiClimate_security: 1.304 → 1.205; wikiHuman-wildlife_conflict: 0.8333 → 0.7634
- `1.2 slow climate onset` (7 of 7 documents): examplefile: 33.33 → 41.67; wiki2023_United_Nations_Climate_Change_Conference: 36.89 → 35.56; wikiAnthropocene: 20.82 → 20.78; wikiClimate_change: 51.59 → 51.51; wikiClimate_justice: 57.07 → 54.76; wikiClimate_security: 68.7 → 65.06; wikiHuman-wildlife_conflict: 0.8333 → 0.7634
- `2. climate change and human wild life conflict` (3 of 7 documents): wikiAnthropocene: 0.8163 → 0.7843; wikiClimate_change: 1.237 → 1.174; wikiClimate_justice: 0.5236 → 0.4762
- `2.1 human-wildlife conflict and fast climate onset` (1 of 7 documents): examplefile: 8.333 → 0
- `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` (1 of 7 documents): examplefile: 8.333 → 0
- `2.2 human-wildlife conflict and slow climate onsets` (4 of 7 documents): examplefile: 8.333 → 16.67; wikiAnthropocene: 0.8163 → 0.7843; wikiClimate_change: 1.237 → 1.174; wikiClimate_justice: 0.5236 → 0.4762
- `3. resource competition and human-wild life conflict` (1 of 7 documents): wikiHuman-wildlife_conflict: 2.5 → 3.053
- `4. increase of adaptive capacity` (2 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 0 → 0.4184; wikiClimate_change: 0.1767 → 0.1678

#### results_document_sentiment.tsv
- `1. climate change` (5 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: 0.1162 → 0.1207; wikiAnthropocene: -0.06166 → -0.055; wikiClimate_change: 0.07591 → 0.07456; wikiClimate_justice: 0.1178 → 0.1145; wikiClimate_security: 0.04519 → 0.03584
- `1.1 fast climate onset` (3 of 7 documents): examplefile: 0.2732 → nan; wikiClimate_change: 0.06157 → 0.06166; wikiClimate_justice: -0.3558 → -0.4816
- `1.2 slow climate onset` (6 of 7 documents): examplefile: 0 → 0.05464; wiki2023_United_Nations_Climate_Change_Conference: 0.1103 → 0.1151; wikiAnthropocene: -0.06166 → -0.055; wikiClimate_change: 0.07748 → 0.07586; wikiClimate_justice: 0.1439 → 0.1405; wikiClimate_security: 0.05724 → 0.04742
- `2.1 human-wildlife conflict and fast climate onset` (1 of 7 documents): examplefile: 0.2732 → nan
- `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` (1 of 7 documents): examplefile: 0.2732 → nan
- `2.2 human-wildlife conflict and slow climate onsets` (1 of 7 documents): examplefile: -0.3182 → -0.0225
- `3. resource competition and human-wild life conflict` (1 of 7 documents): wikiHuman-wildlife_conflict: -0.3176 → -0.1026
- `4. increase of adaptive capacity` (1 of 7 documents): wiki2023_United_Nations_Climate_Change_Conference: nan → 0.9013

#### results_sentences.tsv
- sentences: 1589 → 1692

| document | old sentences | new sentences |
|---|---:|---:|
| examplefile | 12 | 12 |
| wiki2023_United_Nations_Climate_Change_Conference | 225 | 239 |
| wikiAnthropocene | 245 | 255 |
| wikiClimate_change | 566 | 596 |
| wikiClimate_justice | 191 | 210 |
| wikiClimate_security | 230 | 249 |
| wikiHuman-wildlife_conflict | 120 | 131 |

| column | old: sentences True | new: sentences True | change |
|---|---:|---:|---:|
| `fast` | 94 | 95 | +1 |
| `slow` | 671 | 699 | +28 |
| `exclude_climate` | 0 | 0 | +0 |
| `slow1` | 191 | 193 | +2 |
| `slow2` | 723 | 746 | +23 |
| `hwc` | 28 | 55 | +27 |
| `exclude_conflict` | 40 | 41 | +1 |
| `hwc1` | 150 | 151 | +1 |
| `compete2` | 162 | 160 | -2 |
| `resource1` | 221 | 224 | +3 |
| `competition2` | 117 | 118 | +1 |
| `ipcc` | 16 | 16 | +0 |
| `increase` | 211 | 209 | -2 |
| `decrease` | 63 | 63 | +0 |
| `capacity` | 29 | 32 | +3 |
| `co_slow` | 84 | 94 | +10 |
| `co_hwc` | 41 | 41 | +0 |
| `co_resourceandcompetition` | 11 | 13 | +2 |
| `co_increasecapacity` | 3 | 4 | +1 |
| `co_decreasecapacity` | 1 | 1 | +0 |
| `doc_ipcc` | 1244 | 1322 | +78 |
| `sentiment_positive` | 678 | 678 | +0 |
| `sentiment_neutral` | 432 | 531 | +99 |
| `sentiment_neutralpositive` | 1110 | 1209 | +99 |
| `sentiment_negative` | 479 | 483 | +4 |
| `sentiment_neutralnegative` | 911 | 1014 | +103 |
| `a1` | 44 | 41 | -3 |
| `a2` | 698 | 728 | +30 |
| `b1` | 0 | 0 | +0 |
| `b2` | 1 | 0 | -1 |
| `b3` | 11 | 12 | +1 |
| `c1` | 4 | 5 | +1 |
| `d1` | 2 | 3 | +1 |
| `d2` | 1 | 1 | +0 |
| `1. climate change` | 742 | 769 | +27 |
| `1.1 fast climate onset` | 44 | 41 | -3 |
| `1.2 slow climate onset` | 698 | 728 | +30 |
| `2. climate change and human wild life conflict` | 12 | 12 | +0 |
| `2.1 human-wildlife conflict and fast climate onset` | 1 | 0 | -1 |
| `2.1.1 human-wildlife conflict and fast climate onsets in articles that do mention ipcc` | 0 | 0 | +0 |
| `2.1.2 human-wildlife conflict and fast climate onsets in articles that mention ipcc` | 1 | 0 | -1 |
| `2.2 human-wildlife conflict and slow climate onsets` | 11 | 12 | +1 |
| `3. resource competition and human-wild life conflict` | 4 | 5 | +1 |
| `4. increase of adaptive capacity` | 2 | 3 | +1 |
| `5. decrease of adaptive capacity` | 1 | 1 | +0 |

## Regenerating this report

```
python -m ctrlfindr run --config examples/example_project/ctrlfindr.toml
python tools/compare_outputs.py tests/fixtures/golden_v2.1 examples/example_project/Output --rename vadar_compound=vader_compound
```
