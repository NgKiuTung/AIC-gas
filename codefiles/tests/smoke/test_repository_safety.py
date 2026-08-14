from pathlib import Path

import pytest
from gas_power.data import assert_allowed_data_path
from gas_power.settings import PATHS


def test_required_repository_directories_exist() -> None:
    for relative in ("codefiles/legacy", "codefiles/pipelines", "docs/experimental_docs", "results/registry"):
        assert (PATHS.root / relative).is_dir()


def test_scoring_directory_is_fail_closed_without_accessing_it() -> None:
    forbidden = Path("dataset") / "初赛-评分所用测试集" / "never-open.csv"
    with pytest.raises(PermissionError):
        assert_allowed_data_path(forbidden)


def test_training_path_is_allowed() -> None:
    assert assert_allowed_data_path(Path("dataset") / "初赛-数据集").parts[-1] == "初赛-数据集"
