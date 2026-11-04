"""Loose time budgets that catch accidental slowdowns."""

import time

import numpy as np
import pandas as pd

from conftest import EXAMPLE
from ctrlfindr.engine import analyze_sentences
from ctrlfindr.framework import load_framework


def test_analysis_of_32k_sentences(golden_v21_sentences):
    fw = load_framework(EXAMPLE / "Input" / "Assessment_framework.ods")
    g = pd.concat([golden_v21_sentences] * 20, ignore_index=True)
    inp = pd.DataFrame({
        "document": g["document"] + "_" + (np.arange(len(g)) // len(golden_v21_sentences)).astype(str),
        "text": g["sentences"] + " " + np.arange(len(g)).astype(str),
        "vader_compound": g["vadar_compound"],
    })
    t = time.perf_counter()
    res = analyze_sentences(inp, fw)
    elapsed = time.perf_counter() - t
    assert len(res) == 31780
    assert elapsed < 20, f"analysis took {elapsed:.1f}s"  # ~1 s on a laptop; v2.1 needed ~20 s


def test_long_segments(golden_v21_sentences):
    fw = load_framework(EXAMPLE / "Input" / "Assessment_framework.ods")
    g = golden_v21_sentences
    joined = g.groupby(g.index // 200).agg({"document": "first", "sentences": " ".join, "vadar_compound": "first"})
    inp = pd.DataFrame({"document": joined["document"] + joined.index.astype(str), "text": joined["sentences"],
                        "vader_compound": joined["vadar_compound"]})
    t = time.perf_counter()
    analyze_sentences(inp, fw, long_segment_words=0)
    assert time.perf_counter() - t < 10  # v2.1 needed ~10 s for this
