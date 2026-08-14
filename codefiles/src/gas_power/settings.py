"""Central project paths and non-negotiable data boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT

    @property
    def code(self) -> Path:
        return self.root / "codefiles"

    @property
    def legacy(self) -> Path:
        return self.code / "legacy"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def training_data(self) -> Path:
        return self.root / "dataset" / "初赛-数据集"

    @property
    def scoring_data(self) -> Path:
        """Lexical location only; callers must still pass the submission access gate."""
        return self.root / "dataset" / "初赛-评分所用测试集"


PATHS = ProjectPaths()
