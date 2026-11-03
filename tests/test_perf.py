"""Loose time budgets that catch accidental slowdowns (v2.1 was 10-100x slower)."""

import time

import numpy as np
import pandas as pd

from ctrlfindr.analyze import analyze_sentences
from ctrlfindr.config import SentimentSettings
from ctrlfindr.framework import load_framework


def test_analysis_of_32k_sentences(golden_v21_sentences, example_framework_path):
    fw = load_framework(example_framework_path)
    g = pd.concat([golden_v21_sentences] * 20, ignore_index=True)
    inp = pd.DataFrame({
        "document": g["document"] + "_" + (np.arange(len(g)) // len(golden_v21_sentences)).astype(str),
        "text": g["sentences"] + " " + np.arange(len(g)).astype(str),  # unique sentences
        "vader_compound": g["vadar_compound"],
    })
    t = time.perf_counter()
    res = analyze_sentences(inp, fw, SentimentSettings())
    elapsed = time.perf_counter() - t
    assert len(res) == 31780
    assert elapsed < 20, f"analysis took {elapsed:.1f}s"  # ~1 s on a laptop; v2.1 needed ~20 s
