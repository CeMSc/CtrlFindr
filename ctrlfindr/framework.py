"""Load and validate an assessment framework (.ods or .xlsx), format version 1.

Sheets (names are case-insensitive; columns marked * are required):

meta         key*, value                  format_version (required), title, version, author,
                                          language, description, and any other key
settings     key*, value, note            coding settings (optional sheet; empty value = default)
taxonomy     list*, term*, note           one row per term; "-phrase" / "-@list" exclude
queries      name*, query*, note          named expressions (see ctrlfindr.query)
variables    number*, name*, definition, parent, note
tests        sentence*, expect_true, expect_false, group, note   (optional)
instructions (ignored)

Everything is checked before any text is processed and all problems are reported together,
with sheet and row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import query as q
from .common import ValidationError, collapse_ws, normalize_text, sha256_file
from .matching import ListSpec, TermError, parse_term
from .spreadsheet import SUFFIXES, read_workbook

FORMAT_VERSION = "1"
SENTIMENT_TERMS = (
    "sentiment_positive",
    "sentiment_neutral",
    "sentiment_neutralpositive",
    "sentiment_negative",
    "sentiment_neutralnegative",
)
OUTPUT_COLUMNS = {"document", "sentences", "sentence_id", "structure", "start", "end", "vader_compound"}
STRUCTURE_LABELS = ("header", "body", "list", "table", "footnote", "unstructured")  # see ctrlfindr.ingest
RESERVED = OUTPUT_COLUMNS | set(SENTIMENT_TERMS)

SHEETS = {
    "meta": (("key",), ("value",)),
    "settings": (("key",), ("value", "note")),
    "taxonomy": (("list", "term"), ("note",)),
    "queries": (("name", "query"), ("note",)),
    "variables": (("number", "name"), ("definition", "parent", "note")),
    "tests": (("sentence",), ("expect_true", "expect_false", "group", "note")),
}
REQUIRED_SHEETS = ("meta", "taxonomy", "queries", "variables")

_VAR_NUMBER = re.compile(r"\d+(\.\d+)*")


@dataclass
class Settings:
    """Coding settings stored in the framework (sheet 'settings')."""

    sentiment: bool = False
    sentiment_positive_threshold: float = 0.05
    sentiment_negative_threshold: float = -0.05
    aggregate_children: bool = False
    tests_must_pass: bool = True
    exclude_sentence_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    name: str
    text: str
    ast: q.Node
    note: str = ""


@dataclass
class Variable:
    number: str
    name: str
    text: str
    ast: q.Node | None
    parent: str | None
    note: str = ""
    children: list[str] = field(default_factory=list)

    @property
    def column(self) -> str:
        """Output column name: ``"1. name"``, ``"1.1 name"``."""
        num = self.number if "." in self.number else self.number + "."
        return f"{num} {self.name}"


@dataclass(frozen=True)
class TestCase:
    row: int
    group: str
    sentence: str
    expect_true: tuple[str, ...]
    expect_false: tuple[str, ...]
    note: str = ""


@dataclass
class Framework:
    path: Path
    sha256: str
    meta: dict[str, str]
    settings: Settings
    lists: dict[str, ListSpec]            # named lists, in sheet order
    inline: dict[str, ListSpec]           # {term} used in expressions, keyed "{term}"
    notes: dict[str, str]                 # notes of lists (first note of the list) and queries
    term_notes: dict[tuple[str, str], str]
    queries: list[Query]
    query_order: list[str]
    variables: list[Variable]
    variable_order: list[str]             # variable numbers in evaluation order
    tests: list[TestCase]
    set_lists: dict[int, tuple[str, ...]]  # id(set argument node) -> lists whose matches it uses

    @property
    def all_lists(self) -> dict[str, ListSpec]:
        return {**self.lists, **self.inline}

    def variable(self, number: str) -> Variable:
        return next(v for v in self.variables if v.number == number)


def clean_cell(value: str) -> str:
    """Normalize a cell like the corpus text (NFKC, dash/quote folding, single spaces)."""
    return collapse_ws(normalize_text(value))


def _records(sheet: str, rows: list[list[str]], problems: list[str]) -> list[tuple[int, dict[str, str]]]:
    """Rows of a sheet as (spreadsheet row number, {column: cleaned value}).

    Fully empty rows are skipped; rows missing a required value are reported.
    """
    required, optional = SHEETS[sheet]
    if not rows:
        problems.append(f"sheet '{sheet}' has no header row (expected columns: {', '.join(required + optional)})")
        return []
    header = [clean_cell(h).lower() for h in rows[0]]
    missing = [c for c in required if c not in header]
    if missing:
        problems.append(f"sheet '{sheet}': missing column(s) {', '.join(repr(m) for m in missing)}")
        return []
    index = {c: header.index(c) for c in required + optional if c in header}
    out = []
    for i, row in enumerate(rows[1:], start=2):
        rec = {c: clean_cell(row[j]) if j < len(row) else "" for c, j in index.items()}
        if not any(rec.values()):
            continue
        empty = [c for c in required if not rec[c]]
        if empty:
            problems.append(f"sheet '{sheet}', row {i}: empty cell(s) in {', '.join(repr(e) for e in empty)}")
            continue
        for c in optional:
            rec.setdefault(c, "")
        out.append((i, rec))
    return out


def _bool(value: str) -> bool | None:
    v = value.lower()
    return True if v in ("true", "yes", "1") else False if v in ("false", "no", "0") else None


def _settings(records, problems: list[str]) -> Settings:
    s = Settings()
    seen: dict[str, int] = {}
    for i, rec in records:
        key, value = rec["key"].lower(), rec["value"]
        where = f"sheet 'settings', row {i}"
        if not value:  # left empty: keep the default
            continue
        if key in seen:
            problems.append(f"{where}: setting {key!r} is already set in row {seen[key]}")
            continue
        seen[key] = i
        if not hasattr(s, key):
            problems.append(f"{where}: unknown setting {key!r} (known: {', '.join(Settings.__dataclass_fields__)})")
            continue
        default = getattr(s, key)
        if isinstance(default, tuple):
            items = tuple(dict.fromkeys(x.strip().lower() for x in re.split(r"[,;]", value) if x.strip()))
            unknown = [x for x in items if x not in STRUCTURE_LABELS]
            if unknown:
                problems.append(f"{where}: {key}: unknown sentence type(s) {', '.join(map(repr, unknown))} "
                                f"(known: {', '.join(STRUCTURE_LABELS)})")
                continue
            setattr(s, key, items)
        elif isinstance(default, bool):
            b = _bool(value)
            if b is None:
                problems.append(f"{where}: {key} must be true or false, not {value!r}")
                continue
            setattr(s, key, b)
        else:
            try:
                setattr(s, key, float(value.replace(",", ".")))
            except ValueError:
                problems.append(f"{where}: {key} must be a number, not {value!r}")
    if not (-1 <= s.sentiment_negative_threshold <= 0 <= s.sentiment_positive_threshold <= 1):
        problems.append("sheet 'settings': thresholds must satisfy "
                        "-1 <= sentiment_negative_threshold <= 0 <= sentiment_positive_threshold <= 1")
    if s.sentiment:
        from . import sentiment

        if not sentiment.available():
            problems.append(f"sheet 'settings': {sentiment.MISSING}")
    return s


def _parent_of(number: str) -> str | None:
    return number.rsplit(".", 1)[0] if "." in number else None


def load_framework(path: str | Path) -> Framework:
    """Read and validate a framework; raises ValidationError listing every problem."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError("Assessment framework not found", [f"file does not exist: {path}"])
    if path.suffix.lower() not in SUFFIXES:
        raise ValidationError("Assessment framework has an unsupported file type",
                              [f"{path.name}: use .ods (LibreOffice) or .xlsx (Excel)"])
    try:
        raw = read_workbook(path)
    except Exception as exc:
        raise ValidationError("Assessment framework could not be read", [f"{path}: {exc}"]) from exc
    sheets = {clean_cell(k).lower(): v for k, v in raw.items()}
    title = f"Assessment framework {path.name} is invalid"

    # --- format version first: without it nothing else can be interpreted (R2)
    problems: list[str] = []
    meta: dict[str, str] = {}
    if "meta" not in sheets:
        raise ValidationError(title, ["sheet 'meta' is missing, so this is not a CtrlFindr framework of format "
                                      f"{FORMAT_VERSION}; start from the template in the template/ folder"])
    for i, rec in _records("meta", sheets["meta"], problems):
        if rec["value"]:  # keys left empty in the template are ignored
            meta[rec["key"].lower()] = rec["value"]
    version = meta.get("format_version", "").removesuffix(".0")
    if not version:
        problems.append("sheet 'meta': the row 'format_version' is missing")
    elif version != FORMAT_VERSION:
        problems.append(f"sheet 'meta': format_version is {version!r}, but this version of CtrlFindr reads "
                        f"framework format {FORMAT_VERSION}")
    if problems:
        raise ValidationError(title, problems)
    lang = meta.get("language", "en").lower()
    if lang not in ("en", "english"):
        problems.append(f"sheet 'meta': language {meta['language']!r} is not supported; CtrlFindr analyzes English (en)")
    for sheet in REQUIRED_SHEETS:
        if sheet not in sheets:
            problems.append(f"sheet '{sheet}' is missing")
    if problems:
        raise ValidationError(title, problems)

    settings = _settings(_records("settings", sheets.get("settings", [["key", "value"]]), problems), problems)

    # --- names: one namespace for lists and queries (R5)
    owners: dict[str, str] = {}

    def claim(name: str, where: str) -> bool:
        problem = q.check_name(name)
        if problem:
            problems.append(f"{where}: {problem}")
            return False
        if name in RESERVED:
            problems.append(f"{where}: name {name!r} is reserved by CtrlFindr")
            return False
        if name in owners:
            problems.append(f"{where}: name {name!r} is already used in {owners[name]}")
            return False
        owners[name] = where
        return True

    # --- taxonomy (long layout)
    lists: dict[str, ListSpec] = {}
    notes: dict[str, str] = {}
    term_notes: dict[tuple[str, str], str] = {}
    first_row: dict[str, int] = {}
    for i, rec in _records("taxonomy", sheets["taxonomy"], problems):
        name, term = rec["list"].lower(), rec["term"].lower()
        where = f"sheet 'taxonomy', row {i}"
        if name not in lists:
            if not claim(name, f"{where} (list)"):
                lists.setdefault(f"\x00invalid{i}", ListSpec([]))
                continue
            lists[name] = ListSpec([])
            first_row[name] = i
        spec = lists[name]
        if rec["note"] and name not in notes:
            notes[name] = rec["note"]
        if term.startswith("-@"):
            target = term[2:].strip()
            if target == name:
                problems.append(f"{where}: list {name!r} cannot exclude itself")
            elif target not in spec.exclude_lists:
                spec.exclude_lists.append(target)
            continue
        body = term[1:].strip() if term.startswith("-") else term
        try:
            parse_term(body)
        except TermError as exc:
            problems.append(f"{where}: {exc}")
            continue
        target_list = spec.exclude_terms if term.startswith("-") else spec.terms
        if body not in target_list:
            target_list.append(body)
        if rec["note"]:
            term_notes[(name, term)] = rec["note"]
    lists = {k: v for k, v in lists.items() if not k.startswith("\x00")}
    for name, spec in lists.items():
        where = f"sheet 'taxonomy', list {name!r} (row {first_row[name]})"
        if not spec.terms:
            problems.append(f"{where}: the list has no terms (only exclusions)")
        for target in spec.exclude_lists:
            if target not in lists:
                problems.append(f"{where}: '-@{target}' refers to a list that does not exist")
    if not lists:
        problems.append("sheet 'taxonomy': no lists defined")

    # --- queries
    parsed: list[tuple[int, str, str, q.Node | None, str]] = []
    for i, rec in _records("queries", sheets["queries"], problems):
        name = rec["name"].lower()
        where = f"sheet 'queries', row {i}"
        claim(name, where)
        try:
            ast: q.Node | None = q.parse(rec["query"])
        except q.QuerySyntaxError as exc:
            problems.append(f"{where}: query {name!r}: {exc}")
            ast = None
        parsed.append((i, name, rec["query"], ast, rec["note"]))
        if rec["note"]:
            notes.setdefault(name, rec["note"])
    query_names = {name for _, name, _, _, _ in parsed}
    query_ast = {name: ast for _, name, _, ast, _ in parsed}
    sentiment_names = set(SENTIMENT_TERMS) if settings.sentiment else set()

    inline: dict[str, ListSpec] = {}
    set_lists: dict[int, tuple[str, ...]] = {}

    def resolve_set(node: q.Node, seen: frozenset[str] = frozenset()) -> tuple[str, ...] | None:
        """Lists whose matches a set argument stands for, or None if it is not a set."""
        if isinstance(node, q.InlineTerm):
            return ("{" + node.text + "}",)
        if isinstance(node, q.Name):
            if node.name in lists:
                return (node.name,)
            if node.name in query_ast and query_ast[node.name] is not None and node.name not in seen:
                return resolve_set(query_ast[node.name], seen | {node.name})
            return None
        if isinstance(node, q.Or):
            parts = [resolve_set(o, seen) for o in node.operands]
            return None if any(p is None for p in parts) else tuple(dict.fromkeys(x for p in parts for x in p))
        return None

    def check_expression(ast: q.Node, where: str, allow_vars: bool, var_numbers: set[str]) -> set[str]:
        """Report unknown names and wrong arguments; return the query names used."""
        for n in q.walk(ast):
            if isinstance(n, q.Name):
                if n.name in SENTIMENT_TERMS and not settings.sentiment:
                    problems.append(f"{where}: {n.name!r} needs sentiment, which is switched off in the settings sheet")
                elif n.name not in lists and n.name not in query_names and n.name not in sentiment_names:
                    problems.append(f"{where}: {n.name!r} is not a list, query or sentiment flag")
            elif isinstance(n, q.InlineTerm):
                if n.text.startswith("-"):
                    problems.append(f"{where}: {{{n.text}}}: exclusions belong in the taxonomy sheet")
                    continue
                try:
                    parse_term(clean_cell(n.text).lower())
                    inline.setdefault("{" + n.text + "}", ListSpec([clean_cell(n.text).lower()]))
                except TermError as exc:
                    problems.append(f"{where}: {exc}")
            elif isinstance(n, q.Call) and n.func == "v":
                if not allow_vars:
                    problems.append(f"{where}: V({n.args[0]}) can only be used in the variables sheet")
                elif n.args[0] not in var_numbers:
                    problems.append(f"{where}: V({n.args[0]}) refers to a variable number that does not exist")
        for func, arg in q.set_arguments(ast):
            lists_used = resolve_set(arg)
            if lists_used is None:
                problems.append(
                    f"{where}: {func.upper()} needs word positions, but {q.to_text(arg)!r} is not a list, "
                    "a {term}, or an OR of those (queries count only if they are defined that way)")
            else:
                set_lists[id(arg)] = lists_used
        return q.names(ast) & query_names

    deps: dict[str, set[str]] = {}
    for i, name, _, ast, _ in parsed:
        deps.setdefault(name, set())
        if ast is not None:
            deps[name] |= check_expression(ast, f"sheet 'queries', row {i}: query {name!r}", False, set())
    query_order, cycles = q.dependency_order(deps)
    for cyc in cycles:
        problems.append("sheet 'queries': queries refer to each other in a loop: " + " -> ".join(cyc))

    # --- variables
    v_rows = _records("variables", sheets["variables"], problems)
    numbers: dict[str, int] = {}
    for i, rec in v_rows:
        number = rec["number"].replace(",", ".").rstrip(".")
        if not _VAR_NUMBER.fullmatch(number):
            problems.append(f"sheet 'variables', row {i}: number {rec['number']!r} must look like 1, 1.1 or 1.1.1")
        elif number in numbers:
            problems.append(f"sheet 'variables', row {i}: number {number!r} is already used in row {numbers[number]}")
        else:
            numbers[number] = i
    variables: list[Variable] = []
    for i, rec in v_rows:
        where = f"sheet 'variables', row {i}"
        number = rec["number"].replace(",", ".").rstrip(".")
        if number not in numbers or numbers[number] != i:
            continue
        parent = rec["parent"].replace(",", ".").rstrip(".") or None
        if parent is not None and parent not in numbers:
            problems.append(f"{where}: parent {parent!r} is not a variable number")
            parent = None
        if parent is None:
            implied = _parent_of(number)
            while implied and implied not in numbers:
                implied = _parent_of(implied)
            parent = implied
        if parent == number:
            problems.append(f"{where}: a variable cannot be its own parent")
            parent = None
        ast = None
        if rec["definition"]:
            try:
                ast = q.parse(rec["definition"])
            except q.QuerySyntaxError as exc:
                problems.append(f"{where}: variable {number}: {exc}")
        variables.append(Variable(number, rec["name"], rec["definition"], ast, parent, rec["note"]))
    by_number = {v.number: v for v in variables}
    for v in variables:
        if v.parent in by_number:
            by_number[v.parent].children.append(v.number)
    var_deps: dict[str, set[str]] = {}
    for v in variables:
        where = f"sheet 'variables', row {numbers[v.number]}: variable {v.number}"
        var_deps[v.number] = set()
        if v.ast is not None:
            check_expression(v.ast, where, True, set(numbers))
            var_deps[v.number] |= q.variable_refs(v.ast)
        elif not v.text:
            if not (settings.aggregate_children and v.children):
                problems.append(f"sheet 'variables', row {numbers[v.number]}: variable {v.number} has no definition"
                                + ("" if settings.aggregate_children else
                                   " (it can be left empty only with aggregate_children = true and child variables)"))
        if settings.aggregate_children:
            var_deps[v.number] |= set(v.children)
    # parent chains must not loop
    for v in variables:
        seen, p = {v.number}, v.parent
        while p:
            if p in seen:
                problems.append(f"sheet 'variables': parent relations form a loop at variable {v.number}")
                break
            seen.add(p)
            p = by_number[p].parent if p in by_number else None
    variable_order, vcycles = q.dependency_order(var_deps)
    for cyc in vcycles:
        problems.append("sheet 'variables': variables refer to each other in a loop: " + " -> ".join(cyc))
    if not variables and not problems:
        problems.append("sheet 'variables' defines no variables, so there would be nothing to export")

    # --- tests (R3)
    tests: list[TestCase] = []
    known = set(lists) | query_names | sentiment_names | set(numbers)
    for i, rec in _records("tests", sheets.get("tests", [["sentence"]]), problems):
        where = f"sheet 'tests', row {i}"
        expect = {}
        for col in ("expect_true", "expect_false"):
            items = tuple(x.strip().lower().rstrip(".") if _VAR_NUMBER.fullmatch(x.strip().rstrip(".")) else x.strip().lower()
                          for x in re.split(r"[,;\n]", rec[col]) if x.strip())
            for x in items:
                if x not in known:
                    problems.append(f"{where}: {col} names {x!r}, which is not a list, query, sentiment flag or variable number")
            expect[col] = items
        if not expect["expect_true"] and not expect["expect_false"] and not rec["group"]:
            # rows in a group may be context only (e.g. the sentence that makes DOC(x) true)
            problems.append(f"{where}: the test has no expect_true or expect_false entries (and no group)")
        tests.append(TestCase(i, rec["group"] or f"row {i}", rec["sentence"],
                              expect["expect_true"], expect["expect_false"], rec["note"]))

    # exclusion loops between lists
    if not problems:
        from .matching import Matcher

        try:
            Matcher(lists)
        except TermError as exc:
            problems.append(f"sheet 'taxonomy': {exc}")

    if problems:
        raise ValidationError(title, problems)
    return Framework(
        path=path,
        sha256=sha256_file(path),
        meta=meta,
        settings=settings,
        lists=lists,
        inline=inline,
        notes=notes,
        term_notes=term_notes,
        queries=[Query(name, text, ast, note) for _, name, text, ast, note in parsed],
        query_order=query_order,
        variables=variables,
        variable_order=variable_order,
        tests=tests,
        set_lists=set_lists,
    )
