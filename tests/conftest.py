from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
EXAMPLE = ROOT / "examples" / "example_project"
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def golden_v21_sentences() -> pd.DataFrame:
    """Sentence table committed with CtrlFindr 2.1 (input for analysis-stage tests)."""
    return pd.read_csv(FIXTURES / "golden_v2.1" / "results_sentences.tsv", sep="\t")
