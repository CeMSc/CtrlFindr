"""Project configuration (``ctrlfindr.toml``).

Relative paths are resolved against the folder that contains the config file, so the
same project folder works on Windows and Linux and from any working directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import ValidationError


@dataclass
class Paths:
    text_files: Path = Path("Text_files")
    framework: Path = Path("Input/Assessment_framework.ods")
    output: Path = Path("Output")
    database: Path = Path("Output/analysis.sqlite")


@dataclass
class IngestSettings:
    text_cleaning: bool = True          # remove URLs, collapse "..." to "."
    encoding_errors: str = "strict"     # "strict": stop on non-UTF-8 files; "replace": substitute bad bytes
    long_segment_words: int = 500       # warn about "sentences" longer than this


@dataclass
class SentimentSettings:
    enabled: bool = True
    positive_threshold: float = 0.05    # compound >= this is positive
    negative_threshold: float = -0.05   # compound <= this is negative


@dataclass
class ExportSettings:
    sentence_level: bool = True
    document_level: bool = True


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    sentiment: SentimentSettings = field(default_factory=SentimentSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    source: Path | None = None  # the config file, if loaded from one

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = str(self.source) if self.source else None
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


_SECTIONS = {"paths": Paths, "ingest": IngestSettings, "sentiment": SentimentSettings, "export": ExportSettings}


def _build(raw: dict[str, Any], base: Path, problems: list[str]) -> Config:
    cfg = Config()
    for section, value in raw.items():
        if section not in _SECTIONS:
            problems.append(f"unknown section [{section}] (allowed: {', '.join(_SECTIONS)})")
            continue
        if not isinstance(value, dict):
            problems.append(f"[{section}] must be a table")
            continue
        target = getattr(cfg, section)
        for key, v in value.items():
            if not hasattr(target, key):
                allowed = ", ".join(target.__dataclass_fields__)
                problems.append(f"[{section}] unknown setting {key!r} (allowed: {allowed})")
                continue
            default = getattr(target, key)
            if isinstance(default, Path):
                if not isinstance(v, str):
                    problems.append(f"[{section}] {key} must be a path in quotes")
                    continue
                v = Path(v)
            elif isinstance(default, bool):
                if not isinstance(v, bool):
                    problems.append(f"[{section}] {key} must be true or false")
                    continue
            elif isinstance(default, (int, float)):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    problems.append(f"[{section}] {key} must be a number")
                    continue
                if isinstance(default, int) and v != int(v):
                    problems.append(f"[{section}] {key} must be a whole number")
                    continue
                v = type(default)(v)
            setattr(target, key, v)
    for key in cfg.paths.__dataclass_fields__:
        p = getattr(cfg.paths, key)
        if not p.is_absolute():
            setattr(cfg.paths, key, (base / p).resolve())
    if cfg.ingest.encoding_errors not in ("strict", "replace"):
        problems.append('[ingest] encoding_errors must be "strict" or "replace"')
    s = cfg.sentiment
    if not (-1 <= s.negative_threshold <= 0 <= s.positive_threshold <= 1):
        problems.append("[sentiment] thresholds must satisfy -1 <= negative_threshold <= 0 <= positive_threshold <= 1")
    return cfg


def load_config(path: str | Path) -> Config:
    """Read a TOML config file; raises ValidationError listing every problem."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError("Configuration not found", [f"file does not exist: {path}"])
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValidationError(f"Configuration {path.name} is not valid TOML", [str(exc)]) from exc
    problems: list[str] = []
    cfg = _build(raw, path.parent.resolve(), problems)
    if problems:
        raise ValidationError(f"Configuration {path.name} is invalid", problems)
    cfg.source = path.resolve()
    return cfg


def config_from_dict(raw: dict[str, Any], base: str | Path = ".") -> Config:
    """Build a config from a dict with the same layout as the TOML file (for notebooks/tests)."""
    problems: list[str] = []
    cfg = _build(raw, Path(base).resolve(), problems)
    if problems:
        raise ValidationError("Configuration is invalid", problems)
    return cfg
