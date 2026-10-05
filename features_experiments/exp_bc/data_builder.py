"""Build auditable EXP-B/C source tables without changing the Stage2.2 release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE22 = ROOT / "wjt" / "gas_stage22_rebuild"
sys.path.insert(0, str(STAGE22 / "vendor" / "prior" / "vendor" / "stage1"))
from gasstage.quality import collapse_duplicates, numeric_frame

KINDS = ("gas", "gas_holder", "gas_user", "load")
POLICY = {"hard_absolute_limit": 1e20, "hard_negative_below": -1e-6}
SOURCES = (
    ("初赛-参赛者使用", "Pre_", "preliminary_train"),
    ("初赛-评分所用测试集", "Pre_test_", "preliminary_released_eval"),
    ("复赛-参赛者使用", "Semi_", "semifinal_train"),
)


def read_source(base: Path, directory: str, prefix: str) -> tuple[pd.DataFrame, dict]:
    """Apply the published lexical numeric policy; never impute missing or zero."""
    frames: list[pd.DataFrame] = []
    details: dict = {}
    for kind in KINDS:
        path = base / directory / f"{prefix}{kind}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "datetime" not in raw or raw.columns.duplicated().any():
            raise ValueError(f"Invalid schema: {path}")
        raw.index = pd.DatetimeIndex(
            pd.to_datetime(raw.pop("datetime"), format="%Y-%m-%d %H:%M:%S"),
            name="datetime",
        )
        if raw.index.has_duplicates:
            raise ValueError(f"Duplicate time within source: {path}")
        clean, _, changes, profile = numeric_frame(raw, POLICY)
        frames.append(clean)
        details[kind] = {
            "path": str(path),
            "rows": len(raw),
            "first": str(raw.index.min()),
            "last": str(raw.index.max()),
            "invalid_cells": len(changes),
            "missing_cells": int(sum(item["missing"] for item in profile)),
        }
    joined = pd.concat(frames, axis=1).sort_index()
    if joined.columns.duplicated().any():
        raise ValueError(f"Duplicate column in {directory}")
    return joined, details


def build(base: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    pre_train, pre_train_audit = read_source(base, *SOURCES[0][:2])
    pre_eval, pre_eval_audit = read_source(base, *SOURCES[1][:2])
    semi, semi_audit = read_source(base, *SOURCES[2][:2])
    if list(pre_train.columns) != list(pre_eval.columns) or list(
        pre_train.columns
    ) != list(semi.columns):
        raise ValueError("The three phase schemas differ; do not silently align them")

    # The released evaluation begins at the training endpoint. Stage1 accepts
    # the overlap only if every field agrees, and retains one row.
    pre, duplicates = collapse_duplicates(
        pd.concat([pre_train, pre_eval]).sort_index(kind="stable")
    )
    pre = pre.copy()
    pre["dataset_phase"] = np.where(
        pre.index < pre_eval.index.min(),
        "preliminary_train",
        "preliminary_released_eval",
    )
    semi = semi.copy()
    semi["dataset_phase"] = "semifinal_train"
    pre.to_parquet(output / "preliminary_clean_15m.parquet")
    semi.to_parquet(output / "semifinal_clean_1m.parquet")

    report = {
        "sources": {
            "preliminary_train": pre_train_audit,
            "preliminary_released_eval": pre_eval_audit,
            "semifinal_train": semi_audit,
        },
        "columns": list(pre_train.columns),
        "pre_rows": len(pre),
        "semi_rows": len(semi),
        "overlap": duplicates,
        "cleaning": (
            "Stage1 numeric_frame: invalid parse, nonfinite, abs>1e20, "
            "negative<-1e-6 isolated as NaN; zero retained; no interpolation"
        ),
        "pre_label_semantics": (
            "same-record 15min sample; interval-mean meaning unverified; proxy only"
        ),
        "semi_label_semantics": (
            "1min observed load; future official labels require complete "
            "15/15 block means"
        ),
    }
    audit_path = base / "results" / "exp_bc" / "source_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    args = parser.parse_args()
    report = build(args.base, args.output)
    print(
        json.dumps(
            {k: report[k] for k in ("pre_rows", "semi_rows", "overlap")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
