"""The query language used in the queries and variables sheets: tokenizer and parser.

    expr      := and_expr (OR and_expr)*
    and_expr  := not_expr (AND not_expr)*
    not_expr  := NOT not_expr | primary
    primary   := name | "quoted name" | {term} | ( expr ) | call [compare]
    call      := NEAR(set, set, n) | BEFORE(set, set, n) | NOT_NEAR(set, set, n)
               | DOC(expr) | DOC_SHARE(expr) | WINDOW(expr, n)
               | COUNT(set) | COUNT_DISTINCT(set) | V(variable number)
    compare   := (>= | > | = | != | <= | <) number      (after DOC, DOC_SHARE, COUNT,
                                                        COUNT_DISTINCT only)

Keywords are case-insensitive. ``not`` binds tighter than ``and``, ``and`` tighter than
``or``. A *set* is something with word positions: a list, a {term}, or an OR of those
(also through a query defined that way). The evaluation lives in ctrlfindr.engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")
OPERATORS = {"and", "or", "not"}
# function name -> (argument kinds, may be followed by a comparison)
FUNCTIONS: dict[str, tuple[tuple[str, ...], bool]] = {
    "near": (("set", "set", "int"), False),
    "before": (("set", "set", "int"), False),
    "not_near": (("set", "set", "int"), False),
    "doc": (("expr",), True),
    "doc_share": (("expr",), True),
    "window": (("expr", "int"), False),
    "count": (("set",), True),
    "count_distinct": (("set",), True),
    "v": (("varnum",), False),
}
RESERVED_WORDS = OPERATORS | set(FUNCTIONS)
COMPARATORS = (">=", "<=", "!=", "=", ">", "<")


class QuerySyntaxError(ValueError):
    pass


# --------------------------------------------------------------------------- AST


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class InlineTerm:
    text: str


@dataclass(frozen=True)
class Not:
    operand: "Node"


@dataclass(frozen=True)
class And:
    operands: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    operands: tuple["Node", ...]


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple  # Nodes, ints or a variable-number string, as the signature says
    op: str | None = None
    value: float | None = None


Node = Union[Name, InlineTerm, Not, And, Or, Call]


def check_name(name: str) -> str | None:
    """Problem with a name defined in the framework, or None if it is valid."""
    if not NAME_RE.fullmatch(name):
        hint = re.sub(r"[^a-z0-9_-]+", "_", name).strip("_")
        if hint and not hint[0].isalpha():
            hint = "x_" + hint
        return (f"name {name!r} must start with a letter and contain only letters, digits, '_' or '-'"
                + (f" (for example {hint!r})" if hint else ""))
    if name in RESERVED_WORDS:
        return f"name {name!r} is a reserved word of the query language"
    return None


# --------------------------------------------------------------------- tokenizer

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | "(?P<quoted>[^"]*)"
  | \{(?P<term>[^{}]*)\}
  | (?P<cmp>>=|<=|!=|=|>|<)
  | (?P<num>\d+(?:\.\d+)*)
  | (?P<ident>[A-Za-z][A-Za-z0-9_-]*)
  | (?P<punct>[(),])
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        kind = m.lastgroup
        if kind == "quoted":
            name = m.group("quoted").strip()
            if not name:
                raise QuerySyntaxError(f'empty name "" at position {pos + 1}')
            tokens.append(("name", name.lower(), pos))
        elif kind == "term":
            term = " ".join(m.group("term").split()).lower()
            if not term:
                raise QuerySyntaxError(f"empty term {{}} at position {pos + 1}")
            tokens.append(("term", term, pos))
        elif kind == "cmp":
            tokens.append(("cmp", m.group(), pos))
        elif kind == "num":
            tokens.append(("num", m.group(), pos))
        elif kind == "ident":
            word = m.group().lower()
            if word in OPERATORS:
                tokens.append(("op", word, pos))
            elif word in FUNCTIONS and _next_char(text, m.end()) == "(":
                tokens.append(("func", word, pos))
            else:
                tokens.append(("name", word, pos))
        elif kind == "punct":
            tokens.append((m.group(), m.group(), pos))
        elif kind == "other":
            ch = m.group()
            if ch == '"':
                raise QuerySyntaxError(f"unclosed quotation mark at position {pos + 1}")
            if ch in "{}":
                raise QuerySyntaxError(f"unmatched {ch!r} at position {pos + 1}")
            raise QuerySyntaxError(f"unexpected character {ch!r} at position {pos + 1}")
        pos = m.end()
    return tokens


def _next_char(text: str, pos: int) -> str:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return text[pos] if pos < len(text) else ""


# ------------------------------------------------------------------------ parser


class _Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, what: str):
        tok = self.peek()
        if tok is None or tok[0] != kind:
            where = f"at position {tok[2] + 1}" if tok else "at the end"
            raise QuerySyntaxError(f"expected {what} {where}")
        return self.take()

    def at(self, kind: str, value: str | None = None) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == kind and (value is None or tok[1] == value)

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("the expression is empty")
        node = self.expr()
        tok = self.peek()
        if tok is not None:
            raise QuerySyntaxError(f"unexpected {tok[1]!r} at position {tok[2] + 1}")
        return node

    def expr(self) -> Node:
        items = [self.and_expr()]
        while self.at("op", "or"):
            self.take()
            items.append(self.and_expr())
        return items[0] if len(items) == 1 else Or(tuple(items))

    def and_expr(self) -> Node:
        items = [self.not_expr()]
        while self.at("op", "and"):
            self.take()
            items.append(self.not_expr())
        return items[0] if len(items) == 1 else And(tuple(items))

    def not_expr(self) -> Node:
        if self.at("op", "not"):
            self.take()
            return Not(self.not_expr())
        return self.primary()

    def primary(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("the expression ends where a name or '(' was expected")
        kind, value, pos = tok
        if kind == "name":
            self.take()
            return Name(value)
        if kind == "term":
            self.take()
            return InlineTerm(value)
        if kind == "(":
            self.take()
            node = self.expr()
            if not self.at(")"):
                raise QuerySyntaxError(f"missing ')' for '(' at position {pos + 1}")
            self.take()
            return node
        if kind == "func":
            return self.call()
        raise QuerySyntaxError(f"unexpected {value!r} at position {pos + 1}")

    def call(self) -> Node:
        _, func, pos = self.take()
        signature, comparable = FUNCTIONS[func]
        self.expect("(", f"'(' after {func.upper()}")
        args = []
        for k, kind in enumerate(signature):
            if k:
                self.expect(",", f"',' ({func.upper()} takes {len(signature)} arguments)")
            if kind in ("expr", "set"):
                args.append(self.expr())
            elif kind == "int":
                tok = self.expect("num", f"a whole number as argument {k + 1} of {func.upper()}")
                if "." in tok[1]:
                    raise QuerySyntaxError(f"{func.upper()} needs a whole number, not {tok[1]}")
                args.append(int(tok[1]))
            else:  # varnum
                tok = self.expect("num", "a variable number, e.g. V(2.1)")
                args.append(tok[1])
        if not self.at(")"):
            tok = self.peek()
            raise QuerySyntaxError(f"{func.upper()} takes {len(signature)} argument(s); expected ')'"
                                   + (f" at position {tok[2] + 1}" if tok else ""))
        self.take()
        op = value = None
        if self.at("cmp"):
            if not comparable:
                raise QuerySyntaxError(f"{func.upper()}(...) cannot be compared with a number")
            op = self.take()[1]
            tok = self.expect("num", f"a number after {op}")
            if tok[1].count(".") > 1:
                raise QuerySyntaxError(f"invalid number {tok[1]}")
            value = float(tok[1])
        elif comparable:  # bare DOC(x), COUNT(x)...: at least one
            op, value = (">", 0.0) if func == "doc_share" else (">=", 1.0)
        if func == "doc_share" and not 0 <= value <= 1:
            raise QuerySyntaxError("DOC_SHARE compares with a share between 0 and 1, e.g. DOC_SHARE(x) >= 0.05")
        return Call(func, tuple(args), op, value)


def parse(text: str) -> Node:
    """Parse an expression; raises QuerySyntaxError."""
    return _Parser(text).parse()


# ----------------------------------------------------------------------- helpers


def walk(node: Node):
    """Yield every node of the tree."""
    yield node
    if isinstance(node, Not):
        yield from walk(node.operand)
    elif isinstance(node, (And, Or)):
        for op in node.operands:
            yield from walk(op)
    elif isinstance(node, Call):
        for a in node.args:
            if isinstance(a, (Name, InlineTerm, Not, And, Or, Call)):
                yield from walk(a)


def names(node: Node) -> set[str]:
    return {n.name for n in walk(node) if isinstance(n, Name)}


def inline_terms(node: Node) -> set[str]:
    return {n.text for n in walk(node) if isinstance(n, InlineTerm)}


def variable_refs(node: Node) -> set[str]:
    return {n.args[0] for n in walk(node) if isinstance(n, Call) and n.func == "v"}


def set_arguments(node: Node):
    """Yield (function name, argument) for every argument that must be a set."""
    for n in walk(node):
        if isinstance(n, Call):
            for kind, arg in zip(FUNCTIONS[n.func][0], n.args):
                if kind == "set":
                    yield n.func, arg


def to_text(node: Node) -> str:
    """Canonical text of an expression (for messages and the codebook)."""
    if isinstance(node, Name):
        return node.name
    if isinstance(node, InlineTerm):
        return "{" + node.text + "}"
    if isinstance(node, Not):
        return "NOT " + _wrap(node.operand)
    if isinstance(node, (And, Or)):
        sep = " AND " if isinstance(node, And) else " OR "
        return sep.join(_wrap(o) for o in node.operands)
    args = ", ".join(str(a) if isinstance(a, (int, str)) else to_text(a) for a in node.args)
    out = f"{node.func.upper()}({args})"
    default = (">", 0.0) if node.func == "doc_share" else (">=", 1.0)
    if node.op is not None and (node.op, node.value) != default:
        out += f" {node.op} {node.value:g}"
    return out


def _wrap(node: Node) -> str:
    return f"({to_text(node)})" if isinstance(node, (And, Or)) else to_text(node)


def dependency_order(deps: dict[str, set[str]]) -> tuple[list[str], list[list[str]]]:
    """Order names so each comes after the names it uses; also return any cycles."""
    order: list[str] = []
    state: dict[str, int] = {}
    cycles: list[list[str]] = []
    keys = list(deps)

    def visit(name: str, path: list[str]) -> None:
        st = state.get(name)
        if st == 2:
            return
        if st == 1:
            cycles.append(path[path.index(name):] + [name])
            return
        state[name] = 1
        for dep in sorted(deps.get(name, ()), key=lambda d: keys.index(d) if d in keys else -1):
            if dep in deps:
                visit(dep, path + [name])
        state[name] = 2
        order.append(name)

    for name in keys:
        visit(name, [])
    return order, cycles
