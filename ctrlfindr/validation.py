"""Validation against hand coding: draw a sample to code, then report precision and recall.

Workflow:

1. ``python -m ctrlfindr sample --n 50`` writes ``validation_sample.tsv`` to the output
   folder: for each variable, ``n`` sentences CtrlFindr flagged and ``n`` it did not, mixed
   in random order. CtrlFindr's own code is not shown, so the coding stays blind.
2. A person fills the ``human_code`` column with 1 (the sentence is about the variable) or
   0, and saves the file as the project's validation file (``[paths] validation``).
3. Every analysis then writes ``validation_report.tsv`` and ``validation_disagreements.tsv``.

Because flagged and unflagged sentences are sampled separately (stratified), recall is
estimated by weighting each stratum by its size in the corpus:

    precision = TP_sample / n_flagged_sample
    TP_est    = precision * N_flagged
    FN_est    = (FN_sample / n_unflagged_sample) * N_unflagged
    recall    = TP_est / (TP_est + FN_est)

This assumes the coded sentences were drawn at random within each stratum, as ``sample``
does. Coded sentences are matched to the current results by document and sentence_id.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DOCUMENTS, RESULTS, SENTENCES, CtrlFindrError, ValidationError, connect, read_table, table_exists
from .config import Config
from .framework import Framework, load_framework

log = logging.getLogger("ctrlfindr")


def _target_column(fw: Framework, target: str) -> str | None:
    t = str(target).strip().lower().replace(",", ".").rstrip(".")
    numbers = {v.number: v.column for v in fw.variables}
    if t in numbers:
        return numbers[t]
    if t.endswith(".0") and t[:-2] in numbers:  # a spreadsheet saved variable 2 as 2.0
        return numbers[t[:-2]]
    if t in fw.lists or t in {qq.name for qq in fw.queries}:
        return t
    return None


def sample(cfg: Config, n: int = 50, targets: list[str] | None = None, seed: int = 1) -> Path:
    """Write a blind validation sample (n flagged + n unflagged sentences per target)."""
    fw = load_framework(cfg.paths.framework)
    db = cfg.paths.database
    if not db.is_file():
        raise CtrlFindrError(f"database {db} not found; run ingest and analyze first")
    with connect(db) as conn:
        if not all(table_exists(conn, t) for t in (RESULTS, SENTENCES, DOCUMENTS)):
            raise CtrlFindrError("run the analysis first; the sample is drawn from its results")
        results = read_table(conn, RESULTS)
        texts = read_table(conn, SENTENCES)[["document", "sentence_id", "text"]]
    targets = targets or [v.number for v in fw.variables]
    rng = np.random.default_rng(seed)
    rows = []
    for t in targets:
        col = _target_column(fw, t)
        if col is None or col not in results:
            raise CtrlFindrError(f"{t!r} is not a variable number, list or query of the framework")
        flagged = results.index[results[col].astype(bool)].to_numpy()
        unflagged = results.index[~results[col].astype(bool)].to_numpy()
        for pool in (flagged, unflagged):
            take = rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else []
            for i in take:
                rows.append({"document": results.at[i, "document"], "sentence_id": int(results.at[i, "sentence_id"]),
                             "variable": t})
    out = pd.DataFrame(rows).merge(texts, on=["document", "sentence_id"], how="left")
    out = out.rename(columns={"text": "sentence"}).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["human_code"] = ""
    out["note"] = ""
    path = cfg.paths.output / "validation_sample.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    log.info("Wrote %d sentences to code to %s (seed %d)", len(out), path, seed)
    return path


def _code(value) -> int | None:
    v = str(value).strip().lower()
    return 1 if v in ("1", "true", "yes", "y") else 0 if v in ("0", "false", "no", "n") else None


def validation_report(path: Path, results: pd.DataFrame, fw: Framework, out_dir: Path) -> dict:
    """Compare hand codes with the current results; write report and disagreements."""
    try:
        coded = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise ValidationError("Validation file could not be read", [f"{path}: {exc}"]) from exc
    missing = {"document", "sentence_id", "variable", "human_code"} - set(coded.columns)
    if missing:
        raise ValidationError("Validation file is missing columns", [f"{path.name}: {', '.join(sorted(missing))}"])
    coded["code"] = coded["human_code"].map(_code)
    problems = [f"row {i + 2}: human_code {v!r} must be 1 or 0" for i, v in enumerate(coded["human_code"])
                if v.strip() and _code(v) is None]
    if problems:
        raise ValidationError(f"Validation file {path.name} has invalid codes", problems[:20])
    coded = coded[coded["code"].notna()].copy()
    coded["sentence_id"] = pd.to_numeric(coded["sentence_id"], errors="coerce")
    key = results.set_index(["document", "sentence_id"])
    report, disagreements = [], []
    for target, grp in coded.groupby("variable", sort=False):
        col = _target_column(fw, target)
        if col is None:
            raise ValidationError("Validation file names an unknown variable", [f"{target!r}"])
        idx = list(zip(grp["document"], grp["sentence_id"]))
        present = [k in key.index for k in idx]
        grp = grp[present]
        if not len(grp):
            continue
        pred = np.array([bool(key.at[k, col]) for k in zip(grp["document"], grp["sentence_id"])])
        human = grp["code"].to_numpy().astype(int).astype(bool)
        n_flag, n_unflag = int(pred.sum()), int((~pred).sum())
        tp, fp = int((pred & human).sum()), int((pred & ~human).sum())
        fn, tn = int((~pred & human).sum()), int((~pred & ~human).sum())
        N_flag = int(results[col].sum())
        N_unflag = int(len(results) - N_flag)
        precision = tp / n_flag if n_flag else np.nan
        tp_est = precision * N_flag if n_flag else np.nan
        fn_est = (fn / n_unflag) * N_unflag if n_unflag else np.nan
        recall = tp_est / (tp_est + fn_est) if n_flag and n_unflag and (tp_est + fn_est) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and (precision + recall) > 0 else np.nan
        report.append({"variable": target, "column": col, "coded": len(grp), "coded_flagged": n_flag,
                       "coded_unflagged": n_unflag, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "corpus_flagged": N_flag, "corpus_unflagged": N_unflag,
                       "precision": precision, "recall_estimated": recall, "f1": f1})
        for (_, r), p, h in zip(grp.iterrows(), pred, human):
            if p != h:
                disagreements.append({"variable": target, "document": r["document"], "sentence_id": int(r["sentence_id"]),
                                      "ctrlfindr": int(p), "human": int(h), "sentence": r.get("sentence", "")})
    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / "validation_report.tsv", sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(disagreements, columns=["variable", "document", "sentence_id", "ctrlfindr", "human", "sentence"]).to_csv(
        out_dir / "validation_disagreements.tsv", sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    log.info("Validation: %d coded sentences for %d target(s); see validation_report.tsv", int(rep["coded"].sum()) if len(rep) else 0, len(rep))
    return {"file": str(path), "targets": len(rep), "coded_sentences": int(rep["coded"].sum()) if len(rep) else 0}
