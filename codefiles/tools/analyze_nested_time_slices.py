"""Descriptive day/night and intraday diagnostics for nested outer predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "experiments" / "nested_temporal_validation"
INPUT_PATH = RESULT_DIR / "nested_outer_predictions_row_level.csv"
FIGURE_DIR = ROOT / "results" / "figures_safe" / "phase1"
SESSION_LABELS = ["night_00_06", "morning_06_09", "day_09_17", "evening_17_22", "late_22_24"]


def add_time_slices(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["forecast_datetime"] = output["datetime"] + pd.to_timedelta(output["horizon_minutes"], unit="m")
    output["forecast_hour"] = output["forecast_datetime"].dt.hour
    output["forecast_session"] = pd.cut(
        output["forecast_hour"],
        [-1, 5, 8, 16, 21, 23],
        labels=SESSION_LABELS,
    )
    output["day_night"] = np.where(
        output["forecast_hour"].between(7, 18), "day_07_19", "night_19_07"
    )
    return output


def aggregate(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    table = rows.groupby(groups, observed=True, as_index=False).agg(
        samples=("ape_model", "size"),
        model_mape=("ape_model", "mean"),
        persistence_mape=("ape_persistence", "mean"),
    )
    table["absolute_improvement_pp"] = (table["persistence_mape"] - table["model_mape"]) * 100
    return table


def main() -> None:
    rows = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    rows = add_time_slices(rows)
    by_session = aggregate(rows, ["target", "forecast_session"])
    by_day_night = aggregate(rows, ["target", "day_night"])
    by_hour = aggregate(rows, ["target", "forecast_hour"])
    by_session.to_csv(RESULT_DIR / "outer_metrics_by_forecast_session.csv", index=False, encoding="utf-8-sig")
    by_day_night.to_csv(RESULT_DIR / "outer_metrics_by_day_night.csv", index=False, encoding="utf-8-sig")
    by_hour.to_csv(RESULT_DIR / "outer_metrics_by_hour.csv", index=False, encoding="utf-8-sig")

    worst = by_session.sort_values("model_mape", ascending=False).iloc[0]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "descriptive_only_not_used_for_model_selection": True,
        "time_basis": "forecast target datetime = origin datetime + horizon",
        "all_target_sessions_improve": bool((by_session["absolute_improvement_pp"] > 0).all()),
        "worst_target_session": {
            "target": worst["target"],
            "forecast_session": str(worst["forecast_session"]),
            "model_mape": float(worst["model_mape"]),
            "persistence_mape": float(worst["persistence_mape"]),
        },
        "quarter_effect_evaluated": False,
        "quarter_effect_reason": (
            "All preregistered outer pseudo-tests are in April to approximate the May scoring boundary."
        ),
        "external_scoring_data_accessed": False,
    }
    (RESULT_DIR / "time_slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    session_plot = by_session.copy()
    session_plot["model_mape_percent"] = session_plot["model_mape"] * 100
    sns.barplot(
        data=session_plot,
        x="forecast_session",
        y="model_mape_percent",
        hue="target",
        order=SESSION_LABELS,
        ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].set_xlabel("Forecast target session")
    axes[0].set_ylabel("Nested outer MAPE (%)")
    axes[0].set_title("A. Intraday risk by target")
    axes[0].legend(frameon=False)

    for target, target_rows in by_hour.groupby("target"):
        axes[1].plot(
            target_rows["forecast_hour"],
            target_rows["model_mape"] * 100,
            marker="o",
            label=target,
        )
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xlabel("Forecast target hour")
    axes[1].set_ylabel("Nested outer MAPE (%)")
    axes[1].set_title("B. Hourly error profile")
    axes[1].legend(frameon=False)
    fig.suptitle("Nested temporal pseudo-test: day/night and peak-period diagnostics", fontsize=18)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"nested_temporal_time_slices.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
