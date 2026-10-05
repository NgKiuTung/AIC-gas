"""发现初复赛四表并输出可复现的数据、Schema 与质量审计。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRELIMINARY_DIR = PROJECT_ROOT / "初赛-参赛者使用"
SEMIFINAL_DIR = PROJECT_ROOT / "交付_20260924(1)" / "data" / "B_保留缺失"
RAW_SEMIFINAL_DIR = PROJECT_ROOT / "复赛-参赛者使用"
DATASET_DIR = PROJECT_ROOT / "dataset_v3_jan_sep"
DOCS_DIR = PROJECT_ROOT / "docs"
SIGNATURES = {
    "gas": "blast_furnace_1",
    "gas_holder": "blast_furnace_gas_holder_2",
    "gas_user": "blast_furnace_user1",
    "load": "generator_1",
}


def discover_tables(directory: Path) -> dict[str, Path]:
    """按字段签名发现四表，文件名仅供报告展示。"""
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        matches = [key for key, signature in SIGNATURES.items() if signature in columns]
        if len(matches) != 1:
            raise ValueError(f"无法唯一识别数据表: {path.name}, matches={matches}")
        key = matches[0]
        if key in found:
            raise ValueError(f"重复的数据表类别 {key}: {found[key]} / {path}")
        found[key] = path
    if set(found) != set(SIGNATURES):
        raise ValueError(f"目录 {directory} 的四表不完整: {set(SIGNATURES) - set(found)}")
    return found


def read_table(path: Path) -> pd.DataFrame:
    """读取时间戳；重复与缺档交由审计统计，不在此处修复。"""
    frame = pd.read_csv(path, parse_dates=["datetime"])
    if frame["datetime"].isna().any():
        raise ValueError(f"无效 datetime: {path}")
    return frame.sort_values("datetime", kind="stable").reset_index(drop=True)


def longest_run(mask: np.ndarray) -> int:
    """统计布尔序列最长连续真值段。"""
    if not mask.any():
        return 0
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)
    return int((ends - starts).max())


def table_timing(frame: pd.DataFrame) -> dict[str, object]:
    """记录真实时间粒度、缺点与重复，不根据文件名推断。"""
    times = frame["datetime"]
    differences = times.diff().dropna()
    nominal = differences[differences > pd.Timedelta(0)].mode().iloc[0]
    grid = pd.date_range(times.min(), times.max(), freq=nominal)
    missing = grid.difference(pd.DatetimeIndex(times))
    return {
        "start": str(times.min()),
        "end": str(times.max()),
        "rows": len(frame),
        "columns": len(frame.columns),
        "nominal_interval": str(nominal),
        "duplicate_timestamps": int(times.duplicated().sum()),
        "missing_timestamps": len(missing),
        "missing_examples": [str(value) for value in missing[:8]],
    }


def quality_rows(
    phase: str, key: str, frame: pd.DataFrame, nominal_minutes: int
) -> list[dict[str, object]]:
    """逐字段记录缺失、零值、异常尺度与连续段。"""
    rows: list[dict[str, object]] = []
    for column in frame.columns.drop("datetime"):
        values = pd.to_numeric(frame[column], errors="coerce")
        array = values.to_numpy(dtype=float)
        finite = values[np.isfinite(array)]
        quantiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        q1 = float(quantiles.loc[0.25]) if len(finite) else np.nan
        q3 = float(quantiles.loc[0.75]) if len(finite) else np.nan
        spread = q3 - q1
        extreme = (
            int(((finite < q1 - 3 * spread) | (finite > q3 + 3 * spread)).sum())
            if spread > 0
            else 0
        )
        rows.append(
            {
                "phase": phase,
                "table": key,
                "field": column,
                "dtype": str(frame[column].dtype),
                "rows": len(frame),
                "valid_count": len(finite),
                "missing_ratio": float(values.isna().mean()),
                "zero_ratio": float((values == 0).mean()),
                "inf_count": int(np.isinf(array).sum()),
                "extreme_3iqr_count": extreme,
                "longest_zero_minutes": longest_run(array == 0) * nominal_minutes,
                "longest_missing_minutes": longest_run(np.isnan(array)) * nominal_minutes,
                "min": float(finite.min()) if len(finite) else np.nan,
                "p1": float(quantiles.loc[0.01]) if len(finite) else np.nan,
                "p5": float(quantiles.loc[0.05]) if len(finite) else np.nan,
                "p50": float(quantiles.loc[0.5]) if len(finite) else np.nan,
                "p95": float(quantiles.loc[0.95]) if len(finite) else np.nan,
                "p99": float(quantiles.loc[0.99]) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std()) if len(finite) > 1 else np.nan,
            }
        )
    return rows


def discover_workbooks(directory: Path) -> tuple[Path, Path]:
    """按 48 个电价时段与字典字段结构识别工作簿。"""
    price: Path | None = None
    dictionary: Path | None = None
    for path in directory.glob("*.xlsx"):
        book = pd.ExcelFile(path)
        sheets = [pd.read_excel(book, sheet_name=name) for name in book.sheet_names]
        if any(sheet.shape[0] == 48 and sheet.shape[1] >= 13 for sheet in sheets):
            price = path
        elif any("列名" in sheet.columns for sheet in sheets):
            dictionary = path
    if price is None or dictionary is None:
        raise ValueError("无法按内容识别 price 和 data_dictionary 工作簿")
    return price, dictionary


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [cell(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    preliminary_paths = discover_tables(PRELIMINARY_DIR)
    semifinal_paths = discover_tables(SEMIFINAL_DIR)
    raw_semifinal_paths = discover_tables(RAW_SEMIFINAL_DIR)
    price_path, dictionary_path = discover_workbooks(PRELIMINARY_DIR)
    semifinal_price_path = next(
        (
            path for path in (PROJECT_ROOT / "复赛-参赛者使用").glob("*.xlsx")
            if pd.read_excel(path).shape[:2] == (48, 13)
        ),
        None,
    )
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    tables: dict[str, dict[str, pd.DataFrame]] = {"preliminary": {}, "semifinal": {}}
    timing: dict[str, dict[str, dict[str, object]]] = {"preliminary": {}, "semifinal": {}}
    quality: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for phase, paths in (("preliminary", preliminary_paths), ("semifinal", semifinal_paths)):
        for key, path in paths.items():
            frame = read_table(path)
            tables[phase][key] = frame
            details = table_timing(frame)
            timing[phase][key] = details
            nominal_minutes = int(pd.Timedelta(details["nominal_interval"]).total_seconds() / 60)
            quality.extend(quality_rows(phase, key, frame, nominal_minutes))
            inventory.append(
                {
                    "文件": str(path.relative_to(PROJECT_ROOT)),
                    "时间范围": f"{details['start']} ～ {details['end']}",
                    "行数": details["rows"],
                    "字段数": details["columns"],
                    "时间粒度": details["nominal_interval"],
                    "对应复赛文件": (
                        str(semifinal_paths[key].relative_to(PROJECT_ROOT))
                        if phase == "preliminary"
                        else "自身"
                    ),
                }
            )

    quality_frame = pd.DataFrame(quality)
    quality_frame.to_csv(DATASET_DIR / "quality_by_field.csv", index=False)
    (DATASET_DIR / "timing_audit.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(DATASET_DIR / "source_inventory.csv", index=False)
    raw_comparison_rows: list[dict[str, object]] = []
    raw_quality: list[dict[str, object]] = []
    for key, raw_path in raw_semifinal_paths.items():
        raw = read_table(raw_path)
        cleaned = tables["semifinal"][key].set_index("datetime")
        aligned = raw.set_index("datetime").reindex(cleaned.index)
        common = sorted(set(aligned) & set(cleaned))
        changed = (
            aligned[common].ne(cleaned[common])
            & ~(aligned[common].isna() & cleaned[common].isna())
        )
        raw_quality.extend(quality_rows("semifinal_raw", key, raw, 1))
        raw_comparison_rows.append(
            {
                "表": key,
                "原始行数": len(raw),
                "B版行数": len(cleaned),
                "初赛与原始复赛业务字段一致": (
                    set(tables["preliminary"][key].columns)
                    == set(raw.columns)
                ),
                "原始复赛专属列": ", ".join(sorted(set(raw) - set(tables["preliminary"][key]))),
                "B版专属列": ", ".join(sorted(set(cleaned) - set(aligned))),
                "业务值改动单元格": int(changed.to_numpy().sum()),
            }
        )
    raw_comparison = pd.DataFrame(raw_comparison_rows)
    raw_comparison.to_csv(DATASET_DIR / "raw_vs_b_cleaned.csv", index=False)
    pd.DataFrame(raw_quality).to_csv(DATASET_DIR / "raw_semifinal_quality_by_field.csv", index=False)
    DOCS_DIR.joinpath("preliminary_data_inventory.md").write_text(
        "# 初赛与复赛数据清单\n\n"
        + _markdown_table(inventory_frame)
        + "\n\n初赛辅助文件：`"
        + str(price_path.relative_to(PROJECT_ROOT))
        + "`（48 个半小时时段 × 12 个月）；`"
        + str(dictionary_path.relative_to(PROJECT_ROOT))
        + "`（文件及字段说明）。文件类别按字段签名识别。\n",
        encoding="utf-8",
    )

    pre_fields = {
        column: (key, frame[column].dtype)
        for key, frame in tables["preliminary"].items()
        for column in frame.columns.drop("datetime")
    }
    semi_fields = {
        column: (key, frame[column].dtype)
        for key, frame in tables["semifinal"].items()
        for column in frame.columns.drop("datetime")
    }
    quality_lookup = quality_frame.set_index(["phase", "field"])
    schema_rows: list[dict[str, object]] = []
    for field in sorted(set(pre_fields) | set(semi_fields)):
        in_pre, in_semi = field in pre_fields, field in semi_fields
        pre_median = quality_lookup.loc[("preliminary", field), "p50"] if in_pre else np.nan
        semi_median = quality_lookup.loc[("semifinal", field), "p50"] if in_semi else np.nan
        ratio = semi_median / pre_median if np.isfinite(pre_median) and pre_median != 0 else np.nan
        scale_flag = bool(np.isfinite(ratio) and (ratio > 3 or ratio < 1 / 3))
        schema_rows.append(
            {
                "字段": field,
                "初赛存在": in_pre,
                "复赛存在": in_semi,
                "dtype一致": in_pre and in_semi and pre_fields[field][1] == semi_fields[field][1],
                "单位一致": "无单位元数据，未证实" if in_pre and in_semi else "不适用",
                "含义一致": "名称一致；采样统计口径待证实" if in_pre and in_semi else "阶段专属",
                "可合并": "条件性：保持原粒度" if in_pre and in_semi else "仅作阶段审计",
                "初赛中位数": pre_median,
                "复赛中位数": semi_median,
                "中位数比_复赛除初赛": ratio,
                "尺度警报": scale_flag,
            }
        )
    schema = pd.DataFrame(schema_rows)
    schema.to_csv(DATASET_DIR / "schema_by_field.csv", index=False)
    present = schema.loc[:, ["字段", "初赛存在", "复赛存在", "dtype一致", "单位一致", "含义一致", "可合并"]]
    scale = schema.loc[schema["尺度警报"], ["字段", "初赛中位数", "复赛中位数", "中位数比_复赛除初赛"]]
    DOCS_DIR.joinpath("preliminary_vs_semifinal_schema_audit.md").write_text(
        "# 初赛与复赛 Schema 审计\n\n"
        "## 判断\n\n初赛实测 15 分钟采样，复赛 B 版为 1 分钟采样；初赛字典只写“分钟级”，"
        "未说明 `Pre_load.csv` 是时点负荷还是 15 分钟均值。两阶段不能无损拼成同质 1 分钟观测，"
        "也不能在未确认口径前把初赛负荷直接用于复赛的正式区间均值标签。"
        "原始数值不做比例变换或跨阶段填充。\n\n"
        "## 字段对照\n\n" + _markdown_table(present) + "\n\n"
        "## 数量级警报（中位数比超过 3 倍或低于 1/3）\n\n"
        + (_markdown_table(scale) if not scale.empty else "无。")
        + "\n\n尺度变化可能由工况变化造成，不能凭分位数推断单位变更。"
        "字典没有各字段计量单位与聚合方式；单位一致性保持未证实。"
        "初赛特有的全空字段不得零填，复赛特有的质量标记不能回填初赛或进入模型。\n\n"
        "## 原始复赛与 B 清洗版\n\n" + _markdown_table(raw_comparison) + "\n\n"
        "原始复赛四表与初赛四表的字段结构更接近；B 版新增在线、恒值和填充标记，"
        "并对少量业务值做前向填充。统一清洗使用两阶段原始四表，B 版保留为 5～9 月历史基线参照。\n\n"
        "## 统一策略\n\n"
        "1. 保留初赛 15 分钟原始点与复赛 1 分钟原始点，统一字段名但记录 `dataset_phase` 和实际观测粒度。\n"
        "2. 单独标记 2025-05-01 00:01 至 2025-05-02 23:59 的不可用时段，不跨界插值。\n"
        "3. 复赛的 15 分钟标签由真实一分钟负荷均值构造；初赛标签口径待证实，不能以复制 15 分钟值冒充均值。\n"
        "4. 仅在源字段、单位、统计口径和时间可用性得到核实后建立跨阶段训练契约。\n",
        encoding="utf-8",
    )

    selected = ["generator_1", "generator_all", "blast_furnace_gas_holder_2", "generator_use_blast_furnace_gas"]
    monthly_rows: list[dict[str, object]] = []
    for phase, phase_tables in tables.items():
        for field in selected:
            source = next(frame for frame in phase_tables.values() if field in frame)
            for month, group in source.groupby(source["datetime"].dt.to_period("M")):
                monthly_rows.append(
                    {"phase": phase, "month": str(month), "field": field,
                     "mean": float(group[field].mean()), "median": float(group[field].median()),
                     "std": float(group[field].std()), "rows": len(group)}
                )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(DATASET_DIR / "monthly_distribution.csv", index=False)
    physical_rows: list[dict[str, object]] = []
    for phase, phase_tables in tables.items():
        load = phase_tables["load"]
        physical_rows.append(
            {
                "阶段": phase,
                "generator_all低于generator_1": int(
                    (load["generator_all"] < load["generator_1"]).sum()
                ),
                "generator_1超200MW": int((load["generator_1"] > 200).sum()),
                "generator_all超440MW": int((load["generator_all"] > 440).sum()),
                "负负荷点": int(
                    ((load[["generator_1", "generator_all"]] < 0).any(axis=1)).sum()
                ),
            }
        )
    physical = pd.DataFrame(physical_rows)
    physical.to_csv(DATASET_DIR / "physical_consistency.csv", index=False)
    price_match = "复赛价格表未找到，无法比对"
    if semifinal_price_path is not None:
        pre_price = pd.read_excel(price_path).iloc[:, 1:13].to_numpy(dtype=float)
        semi_price = pd.read_excel(semifinal_price_path).iloc[:, 1:13].to_numpy(dtype=float)
        price_match = (
            "两个阶段的 48×12 电价矩阵逐项完全相同"
            if np.array_equal(pre_price, semi_price)
            else "两个阶段电价矩阵存在差异，须逐项核实"
        )
    timing_table = pd.DataFrame(
        [
            {"阶段": phase, "表": key, "起点": details["start"], "终点": details["end"],
             "间隔": details["nominal_interval"], "重复": details["duplicate_timestamps"],
             "缺失时间点": details["missing_timestamps"]}
            for phase, members in timing.items() for key, details in members.items()
        ]
    )
    key_months = monthly.loc[monthly["field"].isin(["generator_1", "generator_all"])]
    DATASET_DIR.joinpath("data_quality_report.md").write_text(
        "# 初赛与复赛数据质量及工况审计\n\n"
        "## 时间结构\n\n" + _markdown_table(timing_table) + "\n\n"
        "初赛最后一个时间点为 2025-05-01 00:00；复赛首点为 2025-05-03 00:00。"
        "中间 47 小时 59 分钟没有一分钟真实观测，视为不可用边界。"
        "初赛 15 分钟网格缺档见 `timing_audit.json`。\n\n"
        "## 目标负荷逐月分布\n\n" + _markdown_table(key_months) + "\n\n"
        "## 全字段质量\n\n"
        "`quality_by_field.csv` 提供每字段：缺失率、零值率、Inf、3×IQR 极端值数、"
        "最长连续零值/缺失时长、均值、标准差、P1/P5/P50/P95/P99、最小/最大值。"
        "其中复赛为 B 清洗版；原始复赛对应 `raw_semifinal_quality_by_field.csv`，"
        "改值单元格见 `raw_vs_b_cleaned.csv`。极端值是统计提示，不自动判定为传感器异常；"
        "零值不自动视作缺失。\n\n"
        "## 多变量与工业边界核对\n\n" + _markdown_table(physical) + "\n\n"
        "该表只检查负荷层级、额定容量和负值；煤气产耗可能受气柜储存与测量缺口影响，"
        "不能要求逐分钟产耗严格相等。局部时序跳变需结合设备状态和相邻测点判断，"
        "不凭单一统计阈值删除。\n\n"
        "## 电价口径\n\n" + price_match + "。\n\n"
        "## 漂移解释边界\n\n"
        "逐月分布变化只能证实边际分布漂移；`P(Y|X, Month)` 的概念漂移需要统一口径标签和时间验证。"
        "初赛标签口径未确认前，不以该阶段模型分数判断是否保留初赛月份。\n",
        encoding="utf-8",
    )
    print(f"审计完成: {len(inventory)} 个 CSV, {len(schema)} 个字段, {len(quality_frame)} 行质量记录")


if __name__ == "__main__":
    main()
