"""Human-readable codebook of a framework (Markdown), for appendices and review."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from . import query as q
from .framework import Framework


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def codebook(fw: Framework) -> str:
    title = fw.meta.get("title") or fw.path.stem
    out = [f"# Codebook: {_md(title)}", ""]
    for key, value in fw.meta.items():
        if key != "title":
            out.append(f"- **{_md(key)}:** {_md(value)}")
    out.append(f"- **file:** {fw.path.name} (SHA-256 `{fw.sha256[:16]}…`)")
    out += ["", "## Settings", ""]
    out += [f"- `{k}` = {(str(v).lower() if isinstance(v, bool) else ', '.join(v) or '(none)' if isinstance(v, tuple) else v)}"
            for k, v in asdict(fw.settings).items()]

    out += ["", "## Variables", "", "| number | name | definition | parent | note |", "|---|---|---|---|---|"]
    for v in fw.variables:
        definition = q.to_text(v.ast) if v.ast is not None else ""
        if fw.settings.aggregate_children and v.children:
            children = " OR ".join(f"V({c})" for c in v.children)
            definition = f"{definition} OR {children}" if definition else children
        out.append(f"| {v.number} | {_md(v.name)} | `{_md(definition)}` | {v.parent or ''} | {_md(v.note)} |")

    out += ["", "## Queries", "", "| name | query | note |", "|---|---|---|"]
    for qq in fw.queries:
        out.append(f"| `{qq.name}` | `{_md(q.to_text(qq.ast))}` | {_md(qq.note)} |")

    out += ["", "## Lists", ""]
    for name, spec in fw.lists.items():
        out.append(f"### `{name}`" + (f": {_md(fw.notes[name])}" if name in fw.notes else ""))
        out.append("")
        out.append("Terms: " + ", ".join(f"`{t}`" for t in spec.terms))
        excl = [f"`{t}`" for t in spec.exclude_terms] + [f"all terms of `{x}`" for x in spec.exclude_lists]
        if excl:
            out.append("")
            out.append("Ignored inside: " + ", ".join(excl))
        notes = [(t, n) for (lst, t), n in fw.term_notes.items() if lst == name]
        if notes:
            out.append("")
            out += [f"- `{t}`: {_md(n)}" for t, n in notes]
        out.append("")
    if fw.tests:
        out += ["## Tests", "", f"{len(fw.tests)} test sentence(s) in the tests sheet.", ""]
    return "\n".join(out) + "\n"


def write_codebook(fw: Framework, path: Path) -> None:
    path.write_text(codebook(fw), encoding="utf-8", newline="\n")
