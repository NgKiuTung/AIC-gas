"""Fail-closed path policy for competition data access.

Training and experiment code may never receive a scoring-data path.  A future
submission adapter must opt in with an explicit grant and may only read from
the exact scoring directory.  These checks are lexical: they do not stat,
list, open, or hash the protected directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAINING_DIRECTORY_NAME = "初赛-数据集"
SCORING_DIRECTORY_NAME = "初赛-评分所用测试集"


def _contains_named_directory(candidate: Path, directory_name: str) -> bool:
    return directory_name in candidate.parts


def assert_training_data_path(path: str | Path) -> Path:
    """Reject scoring paths in every training, validation, and experiment flow."""
    candidate = Path(path)
    if _contains_named_directory(candidate, SCORING_DIRECTORY_NAME):
        raise PermissionError("Scoring data is forbidden in training and experiment workflows")
    return candidate


@dataclass(frozen=True)
class SubmissionAccessGrant:
    """Explicit capability required by the final, read-only submission adapter."""

    purpose: str
    read_only: bool = True
    allow_scoring_input: bool = False


def assert_submission_input_path(path: str | Path, grant: SubmissionAccessGrant | None) -> Path:
    """Allow a scoring input only with an explicit read-only inference grant."""
    candidate = Path(path)
    if grant is None or not grant.allow_scoring_input:
        raise PermissionError("An explicit scoring-input grant is required")
    if not grant.read_only:
        raise PermissionError("Scoring input access must be read-only")
    if grant.purpose != "final_submission_inference":
        raise PermissionError("Scoring input may only be used for final submission inference")
    if not _contains_named_directory(candidate, SCORING_DIRECTORY_NAME):
        raise PermissionError("Submission adapter may only read the exact scoring directory")
    return candidate


# Backward-compatible name.  It intentionally retains training-only semantics.
assert_allowed_data_path = assert_training_data_path
