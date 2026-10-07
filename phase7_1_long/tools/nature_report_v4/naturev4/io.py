"""Input discovery and parsing for Phase7.1 report figures."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Paths:
    """Authoritative paths for archived Phase7.1 report inputs."""

    repo: Path
    phase7: Path

    @property
    def cache(self) -> Path:
        return self.phase7 / "cache" / "main"

    @property
    def research(self) -> Path:
        return self.phase7 / "research_metadata" / "phase7_1_research_01"

    def first_existing(self, *relative: str) -> Path | None:
        """Return the first existing repository-relative path."""
        for rel in relative:
            p = self.repo / rel
            if p.exists():
                return p
        return None


def read_json(path: Path | None) -> Any:
    """Read UTF-8 JSON when present."""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path | None) -> pd.DataFrame | None:
    """Read CSV when present."""
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def parse_online_metrics(paths: Paths) -> dict[str, float]:
    """Parse frozen online metrics from archived report text, then packaged facts."""
    candidates = [
        paths.phase7 / "ONLINE_RESULT.md",
        paths.repo / "release" / "docs" / "SCORE.md",
    ]
    text = next((p.read_text(encoding="utf-8") for p in candidates if p.exists()), "")
    out: dict[str, float] = {}
    patterns = {
        "total": r"(?:Total|总分)[^0-9]*(60\.0903)",
        "short_score": r"(?:Short[\s\S]{0,220}?Score|short[\s\S]{0,220}?score)[^0-9]*(30\.9525)",
        "long_score": r"(?:Long[\s\S]{0,220}?Score|long[\s\S]{0,220}?score)[^0-9]*(29\.1378)",
        "short_accuracy": r"Short[\s\S]{0,180}?Accuracy[^0-9]*(90\.48)",
        "long_accuracy": r"Long[\s\S]{0,180}?Accuracy[^0-9]*(84\.71)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    if len(out) < 5:
        fallback = Path(__file__).resolve().parents[1] / "resources" / "report_facts.json"
        if fallback.exists():
            facts = json.loads(fallback.read_text(encoding="utf-8")).get("final_online", {})
            for key in ("total", "short_score", "long_score", "short_accuracy", "long_accuracy"):
                if key not in out and key in facts:
                    out[key] = float(facts[key])
    return out


def feature_dictionary(paths: Paths) -> pd.DataFrame | None:
    return read_csv(paths.cache / "features" / "dictionary.csv")


def coverage(paths: Paths) -> pd.DataFrame | None:
    return read_csv(paths.cache / "features" / "coverage.csv")


def numeric_profile(paths: Paths) -> pd.DataFrame | None:
    return read_csv(paths.cache / "audit" / "numeric_profile.csv")


def inventory(paths: Paths) -> pd.DataFrame | None:
    return read_csv(paths.cache / "audit" / "inventory.csv")


def changes(paths: Paths) -> pd.DataFrame | None:
    return read_csv(paths.cache / "audit" / "changes.csv")


def ready(paths: Paths) -> dict[str, Any]:
    return read_json(paths.cache / "ready.json") or {}


def source_audit(paths: Paths) -> dict[str, Any]:
    return read_json(paths.cache / "audit" / "source.json") or {}


def screening(paths: Paths) -> dict[str, Any]:
    return read_json(paths.research / "screening.json") or {}


def confirmation(paths: Paths) -> dict[str, Any]:
    return read_json(paths.research / "confirmation.json") or {}


def selection(paths: Paths) -> dict[str, Any]:
    return read_json(paths.research / "selection.json") or {}


def phase6_manifest(paths: Paths) -> dict[str, Any]:
    p = paths.first_existing(
        "docs/experimental_docs/phase6_xgboost_feature_manifest.json",
        "release/algorithm_code/short_phase6/docs/experimental_docs/phase6_xgboost_feature_manifest.json",
    )
    return read_json(p) or {}
