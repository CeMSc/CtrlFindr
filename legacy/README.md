# CtrlFindr 2.1 (legacy)

This folder keeps the last notebook version of CtrlFindr for reference and citation.
It is not maintained; use the `ctrlfindr` package (see the main README) for new work.

- `v2.1/CtrlFindr.ipynb`: the notebook exactly as released in version 2.1.
- `v2.1/ctrlfindr_v21.py`: the notebook's functions extracted unchanged into a module.
  The test suite uses it to check that `tests/fixtures/golden_v2.1` (the outputs that were
  committed with 2.1) can still be reproduced. Its known bugs are deliberately not fixed;
  see `CHANGELOG.md` and `docs/result_changes.md` for what 3.0 changed.

To run the 2.1 notebook, place it in a folder that contains an `Analysis/` folder with
`Text_files/`, `Input/Assessment_framework.ods` and `Output/`, as the notebook's settings
cell expects. It needs `pandas`, `odfpy` and `nltk`; on NLTK 3.8.2 or later, replace
`nltk.download('punkt')` with `nltk.download('punkt_tab')`.
