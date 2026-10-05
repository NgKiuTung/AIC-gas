"""直接依据原始 CSV 时间戳审计真实采样间隔。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from features_experiments.jan_sep.audit_data import discover_tables

ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    ("初赛训练", ROOT / "初赛-参赛者使用", "15min"),
    ("初赛评测", ROOT / "初赛-评分所用测试集", "15min"),
    ("复赛训练", ROOT / "复赛-参赛者使用", "1min"),
)


def inspect_file(path: Path, expected_freq: str) -> dict[str, object]:
    """保留重复与大间隔证据，不依据配置声明推断数据粒度。"""
    times = pd.read_csv(path, usecols=["datetime"], parse_dates=["datetime"])[
        "datetime"
    ]
    if times.isna().any():
        raise ValueError(f"时间戳解析失败: {path}")
    ordered = times.sort_values().reset_index(drop=True)
    gaps = ordered.diff().dropna().value_counts()
    expected = pd.date_range(ordered.iat[0], ordered.iat[-1], freq=expected_freq)
    absent = expected.difference(pd.DatetimeIndex(ordered))
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": len(times),
        "first": str(ordered.iat[0]),
        "last": str(ordered.iat[-1]),
        "duplicates": int(times.duplicated().sum()),
        "missing_expected": len(absent),
        "missing_examples": ", ".join(str(value) for value in absent[:5]),
        "diff_counts": ", ".join(
            f"{key}×{count}" for key, count in gaps.head(5).items()
        ),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for phase, directory, expected_freq in STAGES:
        for table, path in discover_tables(directory).items():
            rows.append(
                {"phase": phase, "table": table, **inspect_file(path, expected_freq)}
            )
    audit = pd.DataFrame(rows)
    fields = [
        "phase",
        "table",
        "rows",
        "first",
        "last",
        "duplicates",
        "missing_expected",
        "diff_counts",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for record in audit[fields].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in record) + " |")
    preliminary_end = pd.Timestamp(audit.loc[audit.phase.eq("初赛训练"), "last"].max())
    semifinal_start = pd.Timestamp(audit.loc[audit.phase.eq("复赛训练"), "first"].min())
    duplicate_boundary = pd.Timestamp("2025-05-01 00:00:00")
    pre_test = ROOT / "初赛-评分所用测试集" / "Pre_test_load.csv"
    pre_train = ROOT / "初赛-参赛者使用" / "Pre_load.csv"
    overlap = duplicate_boundary in set(
        pd.read_csv(pre_train, usecols=["datetime"], parse_dates=["datetime"])[
            "datetime"
        ]
    ) and duplicate_boundary in set(
        pd.read_csv(pre_test, usecols=["datetime"], parse_dates=["datetime"])[
            "datetime"
        ]
    )
    report = (
        "# 原始文件时间频率审计\n\n"
        "以下间隔由原始 CSV 的 `datetime.diff().value_counts()` 得到；缺时刻仅按各阶段"
        "自身的原生频率计算。\n\n" + "\n".join(lines) + "\n\n"
        f"初赛训练末时刻：`{preliminary_end}`；复赛训练首时刻：`{semifinal_start}`。"
        "两者之间的 5 月 1～2 日由已下发的 `Pre_test_` 四表覆盖，"
        "原始文件层面不存在两天空档。"
        "该段保留为独立来源标记；不插值、不伪造 1 分钟值。\n\n"
        f"`2025-05-01 00:00:00` 同时出现在初赛训练与初赛评测负荷文件：{overlap}。"
        "联合数据对同值重叠记录只保留一份，来源标记为 `preliminary_released_eval`。\n\n"
        "建模策略：`Pre_test_` 15 分钟负荷只作为已下发的历史代理标签，"
        "其区间均值语义未经证实；"
        "复赛 1 分钟负荷按完整的 15/15 观测构造正式区间均值。"
        "5 月 3 日起点可以读取在该时点已可用的 `Pre_test_` 历史，"
        "但特征最大观测时刻不得超过起点。\n"
    )
    (ROOT / "docs" / "actual_frequency_audit.md").write_text(report, encoding="utf-8")
    print(audit[fields].to_string(index=False))


if __name__ == "__main__":
    main()
