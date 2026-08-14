"""Create publication-style prediction diagnostics from training-only OOF data.

Figure contract
---------------
Core conclusion: the cleaning-enhanced dynamic ensemble improves the persistence
baseline modestly overall, while errors remain concentrated in the aggregated
50 MW group, longer horizons, and specific operating periods.
Archetype: quantitative grid.
Backend: Python/matplotlib only.
Exports: editable SVG/PDF plus 300 dpi PNG previews and traceable CSV data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OOF_PATH = ROOT / "results" / "experiments" / "multidepth_ensemble" / "selected_oof_predictions.csv"
METRICS_PATH = ROOT / "results" / "experiments" / "multidepth_ensemble" / "metrics_by_fold_target_horizon.csv"
IMPORTANCE_PATH = ROOT / "results" / "experiments" / "cleaning_feature_importance" / "group_importance.csv"
OUT_DIR = ROOT / "results" / "visualizations" / "prediction"
SOURCE_DIR = OUT_DIR / "source_data"
LOG_DIR = ROOT / "results" / "visualizations" / "logs"

BLUE = "#0F4D92"
BLUE_SOFT = "#8AAFD4"
RED = "#B64342"
TEAL = "#42949E"
NEUTRAL = "#767676"
LIGHT = "#D8D8D8"


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def setup_logger() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prediction_visualizations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "40_prediction_visualizations.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def export_figure(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def make_overview(oof: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame) -> tuple[list[str], dict[str, object]]:
    horizon = metrics.groupby(["target", "horizon_minutes"], as_index=False).agg(
        model_mean=("model_mape", "mean"),
        model_sd=("model_mape", "std"),
        persistence_mean=("persistence_mape", "mean"),
        persistence_sd=("persistence_mape", "std"),
    )
    horizon.to_csv(SOURCE_DIR / "overview_horizon_metrics.csv", index=False, encoding="utf-8-sig")

    hour = oof.assign(hour=oof["datetime"].dt.hour).groupby(["target", "hour"], as_index=False).agg(
        model_mape=("ape_model", "mean"), persistence_mape=("ape_persistence", "mean"), samples=("actual", "size")
    )
    hour["relative_improvement"] = (hour["persistence_mape"] - hour["model_mape"]) / hour["persistence_mape"]
    hour.to_csv(SOURCE_DIR / "overview_hour_metrics.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.28, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    target_styles = {
        "generator_1": (BLUE, "50 MW group"),
        "generator_all": (TEAL, "All units"),
    }
    for target, (color, label) in target_styles.items():
        part = horizon[horizon["target"] == target].sort_values("horizon_minutes")
        x = part["horizon_minutes"].to_numpy()
        y = 100 * part["model_mean"].to_numpy()
        sd = 100 * part["model_sd"].fillna(0).to_numpy()
        base = 100 * part["persistence_mean"].to_numpy()
        ax_a.plot(x, y, color=color, marker="o", lw=1.8, ms=4, label=f"{label}: model")
        ax_a.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0)
        ax_a.plot(x, base, color=color, marker="o", lw=1.0, ms=3, ls="--", alpha=0.52, label=f"{label}: persistence")
    ax_a.axhline(5.0, color=RED, lw=1.0, ls=":", label="5% reference")
    ax_a.set_xlabel("Forecast horizon (min)")
    ax_a.set_ylabel("MAPE (%)")
    ax_a.set_title("Error increases with horizon; 50 MW group is limiting")
    ax_a.grid(axis="y", color=LIGHT, lw=0.6)
    ax_a.legend(ncol=2, loc="upper left", columnspacing=0.8, handlelength=2.0)
    panel_label(ax_a, "a")

    target_summary = oof.groupby("target", as_index=False).agg(
        model_mape=("ape_model", "mean"), persistence_mape=("ape_persistence", "mean")
    )
    target_summary["relative_reduction"] = (
        target_summary["persistence_mape"] - target_summary["model_mape"]
    ) / target_summary["persistence_mape"]
    target_summary.to_csv(SOURCE_DIR / "overview_target_summary.csv", index=False, encoding="utf-8-sig")
    x = np.arange(len(target_summary))
    width = 0.34
    ax_b.bar(x - width / 2, 100 * target_summary["persistence_mape"], width, color=LIGHT, edgecolor=NEUTRAL, label="Persistence")
    bars = ax_b.bar(x + width / 2, 100 * target_summary["model_mape"], width, color=[BLUE, TEAL], label="Model")
    for bar, value in zip(bars, 100 * target_summary["model_mape"]):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{value:.2f}%", ha="center", va="bottom", fontsize=7)
    ax_b.set_xticks(x, ["50 MW group", "All units"])
    ax_b.set_ylabel("MAPE (%)")
    ax_b.set_title("Overall OOF error by target")
    ax_b.grid(axis="y", color=LIGHT, lw=0.6)
    ax_b.legend(loc="upper right")
    panel_label(ax_b, "b")

    heat = hour.pivot(index="target", columns="hour", values="model_mape").loc[["generator_1", "generator_all"]] * 100
    sns.heatmap(
        heat,
        ax=ax_c,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        cbar_kws={"label": "MAPE (%)", "shrink": 0.82},
        linewidths=0.25,
        linecolor="white",
    )
    ax_c.set_yticklabels(["50 MW group", "All units"], rotation=0)
    ax_c.set_xlabel("Target hour")
    ax_c.set_ylabel("")
    ax_c.set_title("Time-of-day error concentration")
    panel_label(ax_c, "c")

    imp = importance.sort_values("importance_sum", ascending=True)
    imp.to_csv(SOURCE_DIR / "overview_feature_group_importance.csv", index=False, encoding="utf-8-sig")
    colors = [BLUE if value >= 0.10 else BLUE_SOFT for value in imp["importance_sum"]]
    ax_d.barh(imp["group"], 100 * imp["importance_sum"], color=colors)
    ax_d.set_xlabel("Gain importance share (%)")
    ax_d.set_title("Feature evidence is distributed across dynamics")
    ax_d.grid(axis="x", color=LIGHT, lw=0.6)
    ax_d.text(-0.17, 1.08, "d", transform=ax_d.transAxes, fontsize=9, fontweight="bold", va="bottom")

    paths = export_figure(fig, "prediction_oof_overview")
    stats = {
        "overall_model_mape": float(oof["ape_model"].mean()),
        "overall_persistence_mape": float(oof["ape_persistence"].mean()),
        "worst_target_hour": hour.loc[hour["model_mape"].idxmax()].to_dict(),
    }
    return paths, stats


def make_representative_day(oof: pd.DataFrame) -> tuple[list[str], dict[str, object]]:
    h120 = oof[oof["horizon_minutes"] == 120].copy()
    h120["date"] = h120["datetime"].dt.date
    daily = h120.groupby(["target", "date"], as_index=False).agg(
        model_mape=("ape_model", "mean"), rows=("actual", "size")
    )
    complete = daily[daily["rows"] >= 90]
    target_day: dict[str, object] = {}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), sharex=False, constrained_layout=True)
    for ax, (target, label, color, panel) in zip(
        axes,
        [
            ("generator_1", "50 MW group", BLUE, "a"),
            ("generator_all", "All units", TEAL, "b"),
        ],
    ):
        days = complete[complete["target"] == target].copy()
        median = days["model_mape"].median()
        selected_date = days.iloc[(days["model_mape"] - median).abs().argmin()]["date"]
        part = h120[(h120["target"] == target) & (h120["date"] == selected_date)].sort_values("datetime")
        target_day[target] = {
            "date": str(selected_date),
            "rows": int(len(part)),
            "model_mape": float(part["ape_model"].mean()),
            "persistence_mape": float(part["ape_persistence"].mean()),
        }
        part.to_csv(SOURCE_DIR / f"representative_day_{target}.csv", index=False, encoding="utf-8-sig")
        ax.plot(part["datetime"], part["actual"], color="#272727", lw=1.5, label="Actual")
        ax.plot(part["datetime"], part["prediction"], color=color, lw=1.35, label="Model")
        ax.plot(part["datetime"], part["current"], color=NEUTRAL, lw=0.9, ls="--", alpha=0.75, label="Persistence")
        ax.set_ylabel("Load (MW)")
        ax.set_title(f"{label}, 120-min horizon — median-error complete day ({selected_date})")
        ax.grid(axis="y", color=LIGHT, lw=0.6)
        ax.legend(ncol=3, loc="upper right")
        ax.tick_params(axis="x", rotation=20)
        panel_label(ax, panel)
    paths = export_figure(fig, "prediction_representative_day")
    return paths, target_day


def make_weekday_hour_heatmaps(oof: pd.DataFrame) -> tuple[list[str], dict[str, object]]:
    work = oof.copy()
    work["weekday"] = work["datetime"].dt.dayofweek
    work["hour"] = work["datetime"].dt.hour
    source = work.groupby(["target", "weekday", "hour"], as_index=False).agg(
        model_mape=("ape_model", "mean"), persistence_mape=("ape_persistence", "mean"), samples=("actual", "size")
    )
    source.to_csv(SOURCE_DIR / "weekday_hour_error.csv", index=False, encoding="utf-8-sig")
    vmax = 100 * source["model_mape"].quantile(0.97)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.9), constrained_layout=True)
    names = [("generator_1", "50 MW group", "a"), ("generator_all", "All units", "b")]
    for ax, (target, label, panel) in zip(axes, names):
        matrix = source[source["target"] == target].pivot(index="weekday", columns="hour", values="model_mape") * 100
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="mako",
            vmin=0,
            vmax=vmax,
            cbar_kws={"label": "MAPE (%)", "shrink": 0.80},
            linewidths=0.15,
            linecolor="white",
        )
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
        ax.set_xlabel("Target hour")
        ax.set_ylabel("")
        ax.set_title(f"{label}: weekday × hour error map")
        panel_label(ax, panel)
    paths = export_figure(fig, "prediction_weekday_hour_heatmap")
    worst = source.loc[source.groupby("target")["model_mape"].idxmax()].to_dict(orient="records")
    return paths, {"worst_cells": worst, "color_scale_cap_percent": float(vmax)}


def main() -> None:
    logger = setup_logger()
    configure_style()
    for path in (OOF_PATH, METRICS_PATH, IMPORTANCE_PATH):
        if "评分所用测试集" in str(path):
            raise RuntimeError("Forbidden external scoring dataset path")
    logger.info("Reading training-only OOF and feature-importance artifacts")
    oof = pd.read_csv(OOF_PATH, parse_dates=["datetime"])
    metrics = pd.read_csv(METRICS_PATH)
    importance = pd.read_csv(IMPORTANCE_PATH)

    overview_paths, overview_stats = make_overview(oof, metrics, importance)
    representative_paths, representative_stats = make_representative_day(oof)
    heatmap_paths, heatmap_stats = make_weekday_hour_heatmaps(oof)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only out-of-fold predictions and training-only feature importance",
        "external_scoring_data_accessed": False,
        "backend": "Python/matplotlib-seaborn only",
        "figure_contract": {
            "core_conclusion": "Dynamic cleaning-enhanced ensemble improves persistence modestly; the 50 MW group, longer horizons, and operating periods remain limiting.",
            "archetype": "quantitative grid",
            "exports": ["editable SVG", "PDF", "300 dpi PNG"],
            "validation_definition": "three chronological OOF folds",
        },
        "figures": overview_paths + representative_paths + heatmap_paths,
        "overview_statistics": overview_stats,
        "representative_day_selection": representative_stats,
        "heatmap_statistics": heatmap_stats,
        "qa": {
            "source_data_exported": True,
            "editable_svg_text": True,
            "colorblind_note": "baseline/model distinguished by line style as well as color where compared",
            "selection_note": "representative day is chosen algorithmically as the complete day closest to median 120-minute OOF error, not hand-picked",
            "review_risk": "ensemble/gate parameters were selected on the same OOF pool; displayed metrics are model-selection estimates, not an untouched final test estimate",
        },
    }
    (OUT_DIR / "visualization_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Created %d figure files", len(summary["figures"]))
    logger.info("Overview OOF MAPE: model=%.4f%%, persistence=%.4f%%", overview_stats["overall_model_mape"] * 100, overview_stats["overall_persistence_mape"] * 100)
    logger.info("Wrote visualizations to %s", OUT_DIR)


if __name__ == "__main__":
    main()
