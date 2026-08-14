from pathlib import Path

import pytest
from gas_power.data.io import (
    SCORING_DIRECTORY_NAME,
    SubmissionAccessGrant,
    assert_submission_input_path,
    assert_training_data_path,
)


def test_training_policy_rejects_scoring_path_without_touching_filesystem() -> None:
    candidate = Path("synthetic_root") / SCORING_DIRECTORY_NAME / "Pre_test_load.csv"
    with pytest.raises(PermissionError):
        assert_training_data_path(candidate)


def test_submission_policy_requires_explicit_read_only_final_inference_grant() -> None:
    candidate = Path("synthetic_root") / SCORING_DIRECTORY_NAME / "Pre_test_load.csv"
    with pytest.raises(PermissionError):
        assert_submission_input_path(candidate, None)
    with pytest.raises(PermissionError):
        assert_submission_input_path(
            candidate, SubmissionAccessGrant("final_submission_inference", read_only=False, allow_scoring_input=True)
        )
    accepted = assert_submission_input_path(
        candidate, SubmissionAccessGrant("final_submission_inference", read_only=True, allow_scoring_input=True)
    )
    assert accepted == candidate


def test_submission_policy_rejects_non_scoring_source() -> None:
    grant = SubmissionAccessGrant("final_submission_inference", allow_scoring_input=True)
    with pytest.raises(PermissionError):
        assert_submission_input_path(Path("synthetic_root") / "初赛-数据集" / "Pre_load.csv", grant)
