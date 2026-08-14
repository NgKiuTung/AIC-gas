"""Fail-closed path checks for competition data access."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_DIRECTORY_NAME = "初赛-评分所用测试集"


def assert_allowed_data_path(path: str | Path) -> Path:
    """Reject a path whose lexical components include the scoring dataset."""
    candidate = Path(path)
    if FORBIDDEN_DIRECTORY_NAME in candidate.parts or FORBIDDEN_DIRECTORY_NAME in str(candidate):
        raise PermissionError("The scoring dataset is forbidden for training and experiment workflows")
    return candidate

