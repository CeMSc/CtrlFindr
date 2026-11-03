"""Load the assessment framework (.ods) and validate it before any text is processed.

All problems are collected and reported together, with the sheet and row they come from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import query as q
from .common import ValidationError, collapse_ws, normalize_text, sha256_file
from .ods import read_ods

SENTIMENT_TERMS = (
    "sentiment_positive",
    "sentiment_neutral",
    "sentiment_neutralpositive",
    "sentiment_negative",
    "sentiment_neutralnegative",
)
# Column names the pipeline itself creates; framework names may not reuse them.
RESERVED = {"document", "sentences", "sentence_id", "start", "end", "vader_compound", *SENTIMENT_TERMS}

SHEETS = {
    "variables": ("variable", "variable number", "aggregation query"),
    "search_strings": ("query name", "query"),
    "co_occurrences": ("name of co-occurrence", "first list", "distance between lists", "second list"),
    "doc_conditionals": ("name of document-level conditional", "list"),
    "taxonomy": (),
}

_VAR_NUMBER = re.compile(r"\d+(\.\d+)*\.?")
_AGGREGATION = re.compile(r'"[^"]+"(\s*,\s*"[^"]+")*')
_HAS_WORD = re.compile(r"\w")


@dataclass(frozen=True)
class CoOccurrence:
    name: str
    first: str
    second: str
    distance: int


@dataclass(frozen=True)
class DocConditional:
    name: str
    list: str


@dataclass(frozen=True)
class Query:
    name: str
    text: str
    ast: q.Node


@dataclass(frozen=True)
class Variable:
    number: str
    name: str
    children: tuple[str, ...]

    @property
    def column(self) -> str:
        """Output column name, as in CtrlFindr 2.x: ``"1. name"``, ``"1.1 name"``."""
        num = self.number if "." in self.number else self.number + "."
        return f"{num} {self.name}"


@dataclass
class Framework:
    path: Path
    sha256: str
    taxonomy: dict[str, list[str]]
    co_occurrences: list[CoOccurrence]
    doc_conditionals: list[DocConditional]
    queries: list[Query]
    query_order: list[str]
    variables: list[Variable]
    uses_sentiment: bool = field(default=False)


def clean_cell(value: str) -> str:
    """Normalize a framework cell the way the corpus is normalized, then lowercase."""
    return collapse_ws(normalize_text(value)).lower()


def _sheet_records(
    name: str, rows: list[list[str]], required: tuple[str, ...], problems: list[str]
) -> list[tuple[int, dict[str, str]]]:
    """Rows of a tabular sheet as (spreadsheet row number, {column: value}).

    Fully blank rows are skipped; partly filled rows are reported.
    """
    if not rows:
        problems.append(f"sheet '{name}': header row is missing (expected columns: {', '.join(required)})")
        return []
    header = [clean_cell(h) for h in rows[0]]
    missing = [c for c in required if c not in header]
    if missing:
        problems.append(
            f"sheet '{name}': missing column(s) {', '.join(repr(m) for m in missing)}; "
            "keep the column headers of the template"
        )
        return []
    index = {c: header.index(c) for c in required}
    records = []
    for i, row in enumerate(rows[1:], start=2):
        rec = {c: clean_cell(row[j]) if j < len(row) else "" for c, j in index.items()}
        filled = [c for c, v in rec.items() if v]
        if not filled:
            continue
        if len(filled) < len(required):
            empty = [c for c in required if not rec[c]]
            problems.append(f"sheet '{name}', row {i}: empty cell(s) in {', '.join(repr(e) for e in empty)}")
            continue
        records.append((i, rec))
    return records


def _taxonomy(rows: list[list[str]], problems: list[str]) -> dict[str, list[str]]:
    if not rows:
        problems.append("sheet 'taxonomy' is empty: add at least one list (header = list name)")
        return {}
    header = [clean_cell(h) for h in rows[0]]
    width = max(len(r) for r in rows)
    header += [""] * (width - len(header))
    lists: dict[str, list[str]] = {}
    for j, name in enumerate(header):
        terms: list[str] = []
        for i, row in enumerate(rows[1:], start=2):
            if j < len(row):
                term = clean_cell(row[j])
                if not term:
                    continue
                if not _HAS_WORD.search(term.replace("*", "")):
                    problems.append(f"sheet 'taxonomy', row {i}: term {term!r} has no letters or digits")
                    continue
                if term not in terms:
                    terms.append(term)
        if not name:
            if terms:
                problems.append(f"sheet 'taxonomy', column {j + 1}: terms without a list name in the header row")
            continue
        if name in lists:
            problems.append(f"sheet 'taxonomy': list name {name!r} is used for more than one column")
            continue
        if not terms:
            problems.append(f"sheet 'taxonomy': list {name!r} has no terms (an empty list would match every sentence)")
        lists[name] = terms
    if not lists:
        problems.append("sheet 'taxonomy': no lists found")
    return lists


def load_framework(path: str | Path, sentiment_enabled: bool = True) -> Framework:
    """Read and validate the framework; raises ValidationError listing every problem."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError("Assessment framework not found", [f"file does not exist: {path}"])
    try:
        raw = read_ods(path)
    except Exception as exc:  # zip/xml errors: not an .ods file
        raise ValidationError("Assessment framework is not a readable .ods file", [f"{path}: {exc}"]) from exc
    sheets = {clean_cell(k): v for k, v in raw.items()}
    problems: list[str] = []
    for sheet in SHEETS:
        if sheet not in sheets:
            problems.append(f"sheet '{sheet}' is missing")
    if problems:
        raise ValidationError(f"Assessment framework {path.name} is invalid", problems)

    taxonomy = _taxonomy(sheets["taxonomy"], problems)
    co_rows = _sheet_records("co_occurrences", sheets["co_occurrences"], SHEETS["co_occurrences"], problems)
    dc_rows = _sheet_records("doc_conditionals", sheets["doc_conditionals"], SHEETS["doc_conditionals"], problems)
    q_rows = _sheet_records("search_strings", sheets["search_strings"], SHEETS["search_strings"], problems)
    v_rows = _sheet_records("variables", sheets["variables"], SHEETS["variables"], problems)

    # Every name becomes a column of the same table, so names must be unique across sheets.
    owners: dict[str, str] = {}

    def claim(name: str, where: str) -> None:
        if name in RESERVED:
            problems.append(f"{where}: name {name!r} is reserved by CtrlFindr; choose another name")
        elif name in owners:
            problems.append(f"{where}: name {name!r} is already used in {owners[name]}")
        else:
            owners[name] = where

    for name in taxonomy:
        claim(name, "sheet 'taxonomy'")

    co_occurrences = []
    for i, rec in co_rows:
        where = f"sheet 'co_occurrences', row {i}"
        claim(rec["name of co-occurrence"], where)
        for col in ("first list", "second list"):
            if rec[col] not in taxonomy:
                problems.append(f"{where}: {col} {rec[col]!r} is not a list in the taxonomy sheet")
        try:
            dist_f = float(rec["distance between lists"])
            if dist_f != int(dist_f) or dist_f < 0:
                raise ValueError
            distance = int(dist_f)
        except ValueError:
            problems.append(f"{where}: distance {rec['distance between lists']!r} must be a whole number >= 0")
            distance = 0
        co_occurrences.append(CoOccurrence(rec["name of co-occurrence"], rec["first list"], rec["second list"], distance))

    doc_conditionals = []
    for i, rec in dc_rows:
        where = f"sheet 'doc_conditionals', row {i}"
        claim(rec["name of document-level conditional"], where)
        if rec["list"] not in taxonomy:
            problems.append(f"{where}: list {rec['list']!r} is not a list in the taxonomy sheet")
        doc_conditionals.append(DocConditional(rec["name of document-level conditional"], rec["list"]))

    parsed: list[tuple[int, str, str, q.Node | None]] = []
    for i, rec in q_rows:
        where = f"sheet 'search_strings', row {i}"
        claim(rec["query name"], where)
        try:
            ast: q.Node | None = q.parse(rec["query"])
        except q.QuerySyntaxError as exc:
            problems.append(f"{where}: query {rec['query name']!r}: {exc}")
            ast = None
        parsed.append((i, rec["query name"], rec["query"], ast))

    query_names = [name for _, name, _, _ in parsed]
    base_terms = set(taxonomy) | {c.name for c in co_occurrences} | {d.name for d in doc_conditionals}
    deps: dict[str, set[str]] = {}
    uses_sentiment = False
    for i, name, _, ast in parsed:
        if ast is None:
            deps.setdefault(name, set())
            continue
        used = q.terms(ast)
        for t in sorted(used):
            if t in SENTIMENT_TERMS:
                uses_sentiment = True
                if not sentiment_enabled:
                    problems.append(
                        f"sheet 'search_strings', row {i}: query {name!r} uses {t!r} but sentiment "
                        "analysis is disabled in the configuration"
                    )
            elif t not in base_terms and t not in query_names:
                problems.append(
                    f"sheet 'search_strings', row {i}: query {name!r} refers to {t!r}, which is not a "
                    "taxonomy list, co-occurrence, document-level conditional, sentiment flag or query"
                )
        deps[name] = deps.get(name, set()) | {t for t in used if t in query_names}
    order, cycles = q.dependency_order(deps)
    for cyc in cycles:
        problems.append("sheet 'search_strings': queries refer to each other in a loop: " + " -> ".join(cyc))

    variables = []
    numbers: dict[str, int] = {}
    for i, rec in v_rows:
        where = f"sheet 'variables', row {i}"
        number = rec["variable number"]
        if not _VAR_NUMBER.fullmatch(number):
            problems.append(f"{where}: variable number {number!r} must look like 1, 1.1 or 1.1.1")
        key = number.rstrip(".")
        if key in numbers:
            problems.append(f"{where}: variable number {number!r} is already used in row {numbers[key]}")
        numbers.setdefault(key, i)
        agg = rec["aggregation query"]
        if not _AGGREGATION.fullmatch(agg):
            problems.append(
                f'{where}: aggregation query {agg!r} must be query names in double quotes separated by '
                'commas, e.g. "a1", "a2"'
            )
            children: tuple[str, ...] = ()
        else:
            children = tuple(c.strip() for c in re.findall(r'"([^"]+)"', agg))
            for c in children:
                if c not in query_names:
                    problems.append(f"{where}: {c!r} is not a query name in the search_strings sheet")
        var = Variable(number, rec["variable"], children)
        claim(var.column, where)
        variables.append(var)
    if not variables and not problems:
        problems.append("sheet 'variables' has no variables, so there would be nothing to export")

    if problems:
        raise ValidationError(f"Assessment framework {path.name} is invalid", problems)

    queries = [Query(name, text, ast) for _, name, text, ast in parsed if ast is not None]
    return Framework(
        path=path,
        sha256=sha256_file(path),
        taxonomy=taxonomy,
        co_occurrences=co_occurrences,
        doc_conditionals=doc_conditionals,
        queries=queries,
        query_order=order,
        variables=variables,
        uses_sentiment=uses_sentiment,
    )
