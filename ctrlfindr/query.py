"""The query language of the search_strings sheet: parser and vectorized evaluator.

Grammar (operators are case-insensitive; terms are names in double quotes)::

    expr   := and_expr ("or" and_expr)*
    and_expr := not_expr ("and" not_expr)*
    not_expr := "not" not_expr | atom
    atom   := TERM | "(" expr ")"

Precedence is ``not`` > ``and`` > ``or``, the same as Python, so results equal those of
CtrlFindr 2.x, which evaluated queries with ``eval``. Evaluation never executes code: each
term is looked up in a mapping of boolean NumPy arrays and combined with NumPy operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

# ---------------------------------------------------------------------- AST


@dataclass(frozen=True)
class Term:
    name: str


@dataclass(frozen=True)
class Not:
    operand: "Node"


@dataclass(frozen=True)
class And:
    operands: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    operands: tuple["Node", ...]


Node = Union[Term, Not, And, Or]


class QuerySyntaxError(ValueError):
    pass


# -------------------------------------------------------------------- tokenizer

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<term>"(?P<name>[^"]*)")
  | (?P<lpar>\()
  | (?P<rpar>\))
  | (?P<word>[A-Za-z_]+)
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)
_OPERATORS = {"and", "or", "not"}


def tokenize(query: str) -> list[tuple[str, str, int]]:
    """Split a query into (kind, value, position) tokens; kinds: term, op, (, )."""
    tokens: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(query):
        m = _TOKEN.match(query, pos)
        assert m is not None
        kind = m.lastgroup
        if kind == "term":
            name = m.group("name").strip()
            if not name:
                raise QuerySyntaxError(f'empty term "" at position {pos + 1}')
            tokens.append(("term", name, pos))
        elif kind == "lpar":
            tokens.append(("(", "(", pos))
        elif kind == "rpar":
            tokens.append((")", ")", pos))
        elif kind == "word":
            word = m.group().lower()
            if word not in _OPERATORS:
                raise QuerySyntaxError(
                    f"unexpected word {m.group()!r} at position {pos + 1}; "
                    'names must be inside double quotes, operators are AND, OR, NOT'
                )
            tokens.append(("op", word, pos))
        elif kind == "other":
            ch = m.group()
            if ch == '"':
                raise QuerySyntaxError(f"unclosed quotation mark at position {pos + 1}")
            raise QuerySyntaxError(f"unexpected character {ch!r} at position {pos + 1}")
        pos = m.end()
    return tokens


# ----------------------------------------------------------------------- parser


class _Parser:
    def __init__(self, query: str):
        self.query = query
        self.tokens = tokenize(query)
        self.i = 0

    def peek(self) -> tuple[str, str, int] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str, int]:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_op(self, op: str) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] == op

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("query is empty")
        node = self.expr()
        tok = self.peek()
        if tok is not None:
            raise QuerySyntaxError(f"unexpected {tok[1]!r} at position {tok[2] + 1}")
        return node

    def expr(self) -> Node:
        items = [self.and_expr()]
        while self.at_op("or"):
            self.take()
            items.append(self.and_expr())
        return items[0] if len(items) == 1 else Or(tuple(items))

    def and_expr(self) -> Node:
        items = [self.not_expr()]
        while self.at_op("and"):
            self.take()
            items.append(self.not_expr())
        return items[0] if len(items) == 1 else And(tuple(items))

    def not_expr(self) -> Node:
        if self.at_op("not"):
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("query ends where a term or '(' was expected")
        kind, value, pos = tok
        if kind == "term":
            self.take()
            return Term(value)
        if kind == "(":
            self.take()
            node = self.expr()
            close = self.peek()
            if close is None or close[0] != ")":
                raise QuerySyntaxError(f"missing ')' for '(' at position {pos + 1}")
            self.take()
            return node
        raise QuerySyntaxError(f"unexpected {value!r} at position {pos + 1}")


def parse(query: str) -> Node:
    """Parse a query string into an AST; raises QuerySyntaxError."""
    return _Parser(query).parse()


def terms(node: Node) -> set[str]:
    """All term names referenced by a query."""
    if isinstance(node, Term):
        return {node.name}
    if isinstance(node, Not):
        return terms(node.operand)
    out: set[str] = set()
    for op in node.operands:
        out |= terms(op)
    return out


def evaluate(node: Node, columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate a query over boolean arrays. A missing term raises KeyError."""
    if isinstance(node, Term):
        return np.asarray(columns[node.name], dtype=bool)
    if isinstance(node, Not):
        return ~evaluate(node.operand, columns)
    parts = [evaluate(op, columns) for op in node.operands]
    reduce = np.logical_and.reduce if isinstance(node, And) else np.logical_or.reduce
    return reduce(parts)


def dependency_order(deps: Mapping[str, set[str]]) -> tuple[list[str], list[list[str]]]:
    """Order queries so each comes after the queries it references.

    ``deps`` maps each query name to the query names it uses. Returns the evaluation
    order (stable: sheet order where possible) and the list of cycles found.
    """
    order: list[str] = []
    state: dict[str, int] = {}  # 1 = visiting, 2 = done
    cycles: list[list[str]] = []

    def visit(name: str, path: list[str]) -> None:
        st = state.get(name)
        if st == 2:
            return
        if st == 1:
            cycles.append(path[path.index(name):] + [name])
            return
        state[name] = 1
        for dep in sorted(deps.get(name, ()), key=list(deps).index):
            visit(dep, path + [name])
        state[name] = 2
        order.append(name)

    for name in deps:
        visit(name, [])
    return order, cycles
