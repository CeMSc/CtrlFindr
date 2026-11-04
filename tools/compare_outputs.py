"""Compare two CtrlFindr output folders and print a Markdown report.

    python tools/compare_outputs.py OLD_OUTPUT_DIR NEW_OUTPUT_DIR [--rename old=new ...]

Column names are compared case-insensitively. Document-level tables are compared cell by cell (numbers with a small tolerance).
Sentence-level tables usually have different rows when sentence splitting changed, so
they are compared through per-document sentence counts and per-column totals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOC_TABLES = (
    "results_document",
    "results_document_clipped",
    "results_document_percentage_of_sentences",
    "results_document_sentiment",
)


def read(path: Path) -> pd.DataFrame | None:
    """Read a TSV; column names are lowercased, since 2.1 lowercased variable names."""
    return pd.read_csv(path, sep="\t").rename(columns=str.lower) if path.is_file() else None


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
    return (a.astype(str) == b.astype(str)).to_numpy()


def compare_document_table(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    lines = []
    added = [c for c in new.columns if c not in old.columns]
    removed = [c for c in old.columns if c not in new.columns]
    if added:
        lines.append(f"- columns only in new: {', '.join(f'`{c}`' for c in added)}")
    if removed:
        lines.append(f"- columns only in old: {', '.join(f'`{c}`' for c in removed)}")
    o = old.set_index("document")
    n = new.set_index("document")
    docs = [d for d in o.index if d in n.index]
    if set(o.index) != set(n.index):
        lines.append(f"- documents differ: only old {sorted(set(o.index) - set(n.index))}, only new {sorted(set(n.index) - set(o.index))}")
    changed = []
    for col in [c for c in o.columns if c in n.columns]:
        eq = _equal(o.loc[docs, col], n.loc[docs, col])
        if not eq.all():
            cells = [f"{d}: {o.at[d, col]:.4g} → {n.at[d, col]:.4g}" if isinstance(o.at[d, col], float) or isinstance(n.at[d, col], float)
                     else f"{d}: {o.at[d, col]} → {n.at[d, col]}" for d, e in zip(docs, eq) if not e]
            changed.append((col, cells))
    if not changed:
        lines.append("- all shared cells identical")
    for col, cells in changed:
        lines.append(f"- `{col}` ({len(cells)} of {len(docs)} documents): " + "; ".join(cells))
    return lines


def compare_sentence_table(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    lines = [f"- sentences: {len(old)} → {len(new)}"]
    co, cn = old.groupby("document").size(), new.groupby("document").size()
    lines.append("")
    lines.append("| document | old sentences | new sentences |")
    lines.append("|---|---:|---:|")
    for d in sorted(set(co.index) | set(cn.index)):
        lines.append(f"| {d} | {co.get(d, 0)} | {cn.get(d, 0)} |")
    lines.append("")
    bools = [c for c in old.columns if c in new.columns and old[c].dtype == bool]
    lines.append("| column | old: sentences True | new: sentences True | change |")
    lines.append("|---|---:|---:|---:|")
    for c in bools:
        a, b = int(old[c].sum()), int(new[c].sum())
        lines.append(f"| `{c}` | {a} | {b} | {b - a:+d} |")
    return lines


def compare_dirs(old_dir: Path, new_dir: Path, renames: dict[str, str] | None = None) -> str:
    renames = renames or {}
    out = [f"Comparing `{old_dir}` (old) with `{new_dir}` (new).", ""]
    for name in DOC_TABLES:
        old, new = read(old_dir / f"{name}.tsv"), read(new_dir / f"{name}.tsv")
        out.append(f"### {name}.tsv")
        if old is None or new is None:
            out.append(f"- missing in {'old' if old is None else 'new'}")
        else:
            out += compare_document_table(old.rename(columns=renames), new)
        out.append("")
    old, new = read(old_dir / "results_sentences.tsv"), read(new_dir / "results_sentences.tsv")
    out.append("### results_sentences.tsv")
    if old is None or new is None:
        out.append(f"- missing in {'old' if old is None else 'new'}")
    else:
        out += compare_sentence_table(old.rename(columns=renames), new)
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW", help="treat column OLD in the old output as NEW")
    args = ap.parse_args()
    renames = dict(r.split("=", 1) for r in args.rename)
    print(compare_dirs(args.old, args.new, renames), end="")


if __name__ == "__main__":
    main()
