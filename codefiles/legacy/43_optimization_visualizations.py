"""Publication-style figures for the conservative dispatch backtest.

Core conclusion: the balanced policy shifts forecast-constrained generation
toward higher-price intervals with positive cross-fold uplift, exact resource
conservation, and zero official hard-constraint violations.
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
BACKTEST_DIR = ROOT / "results" / "optimization" / "dispatch_backtest"
VALIDATION_DIR = ROOT / "results" / "optimization" / "dispatch_validation"
OUT_DIR = ROOT / "results" / "visualizations" / "optimization"
SOURCE_DIR = OUT_DIR / "source_data"
LOG_DIR = ROOT / "results" / "visualizations" / "logs"

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GREEN = "#2E9E44"
GOLD = "#D79A20"
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
    logger = logging.getLogger("optimization_visualizations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "43_optimization_visualizations.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def export_figure(fig: plt.Figure, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def make_dispatch_overview(
    representative: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> list[str]:
    representative.to_csv(SOURCE_DIR / "dispatch_representative_window.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(SOURCE_DIR / "dispatch_scenario_tradeoff.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(SOURCE_DIR / "dispatch_selected_by_fold.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.2, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = representative["target_datetime"]
    ax_a.plot(x, representative["baseline_generator_all_mw"], color=NEUTRAL, lw=1.5, marker="o", ms=3.5, label="Forecast baseline")
    ax_a.plot(x, representative["optimized_generator_all_mw"], color=BLUE, lw=2.0, marker="o", ms=4, label="Balanced dispatch")
    ax_a.plot(x, representative["actual_generator_all_mw"], color="#272727", lw=1.0, ls="--", alpha=0.60, label="Historical actual")
    ax_a.set_ylabel("Generation load (MW)")
    ax_a.set_xlabel("Target time")
    ax_a.tick_params(axis="x", rotation=20)
    ax_a.grid(axis="y", color=LIGHT, lw=0.6)
    price_ax = ax_a.twinx()
    price_ax.step(x, representative["price_cny_per_kwh"], where="mid", color=GOLD, lw=1.4, alpha=0.9, label="Tariff")
    price_ax.fill_between(x, 0, representative["price_cny_per_kwh"], step="mid", color=GOLD, alpha=0.08)
    price_ax.set_ylabel("Tariff (CNY/kWh)", color=GOLD)
    price_ax.tick_params(axis="y", colors=GOLD)
    handles1, labels1 = ax_a.get_legend_handles_labels()
    handles2, labels2 = price_ax.get_legend_handles_labels()
    ax_a.legend(handles1 + handles2, labels1 + labels2, ncol=4, loc="upper left")
    ax_a.set_title("Representative price-changing window: load is shifted toward higher tariff")
    panel_label(ax_a, "a", x=-0.06)

    ax_b.axhspan(30_000, 180_000, color="#DDF3DE", alpha=0.65, label="Official safe range")
    ax_b.axhspan(35_000, 175_000, color="#AADCA9", alpha=0.32, label="Balanced internal range")
    initial_holder = float(representative["projected_bfg_holder_m3"].iloc[-1])
    ax_b.axhline(initial_holder, color=NEUTRAL, lw=1.0, ls="--", label="Baseline holder")
    ax_b.plot(x, representative["projected_bfg_holder_m3"], color=GREEN, lw=2.0, marker="o", ms=3.5, label="Projected holder")
    ax_b.set_xlabel("Target time")
    ax_b.set_ylabel("BFG holder (m³)")
    ax_b.tick_params(axis="x", rotation=20)
    ax_b.set_title("Incremental holder trajectory remains inside safety bands")
    ax_b.legend(loc="best", ncol=2)
    ax_b.grid(axis="y", color=LIGHT, lw=0.6)
    panel_label(ax_b, "b")

    colors = {"conservative": TEAL, "balanced": BLUE, "aggressive": RED}
    for row in scenario_summary.itertuples(index=False):
        ax_c.scatter(
            row.mean_absolute_dispatch_change_mw,
            100 * row.mean_revenue_uplift_price_varying,
            s=80 + 1200 * row.normal_minload_transition_fraction,
            color=colors[row.scenario],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax_c.annotate(
            row.scenario.capitalize(),
            (row.mean_absolute_dispatch_change_mw, 100 * row.mean_revenue_uplift_price_varying),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    ax_c.set_xlabel("Mean absolute dispatch change (MW)")
    ax_c.set_ylabel("Mean uplift in price-changing windows (%)")
    ax_c.set_title("Balanced policy retains margin with moderate movement")
    ax_c.grid(color=LIGHT, lw=0.6)
    panel_label(ax_c, "c", x=-0.17, y=1.10)

    paths = export_figure(fig, "optimization_dispatch_overview")

    # Fold stability is kept as a compact independent figure to avoid crowding.
    fig2, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    fold_order = by_fold["fold"].tolist()
    values = 100 * by_fold["mean_uplift_price_varying"].to_numpy()
    bars = ax.bar(fold_order, values, color=[BLUE, TEAL, "#8AAFD4"])
    overall = float(np.average(values, weights=by_fold["price_varying_windows"]))
    ax.axhline(overall, color=RED, lw=1.2, ls="--", label=f"Weighted mean {overall:.2f}%")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean revenue uplift (%)")
    ax.set_title("Balanced dispatch is positive in every chronological fold")
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.legend(loc="upper right")
    paths.extend(export_figure(fig2, "optimization_fold_stability"))
    return paths


def make_uplift_distribution(
    windows: pd.DataFrame,
    schedules: pd.DataFrame,
    bootstrap: pd.DataFrame,
    acceptance: dict[str, object],
) -> list[str]:
    selected = windows[(windows["scenario"] == "balanced") & windows["solver_success"] & (windows["price_levels"] > 1)].copy()
    first_last = schedules[schedules["scenario"] == "balanced"].sort_values("horizon_step").groupby(["fold", "forecast_origin"]).agg(
        start_price=("price_cny_per_kwh", "first"), end_price=("price_cny_per_kwh", "last")
    ).reset_index()
    selected = selected.merge(first_last, on=["fold", "forecast_origin"], validate="one_to_one")
    selected["tariff_transition"] = selected["start_price"].map(lambda value: f"{value:.2f}") + "→" + selected["end_price"].map(lambda value: f"{value:.2f}")
    transitions = selected.groupby("tariff_transition", as_index=False).agg(
        windows=("forecast_origin", "size"),
        mean_uplift=("forecast_revenue_uplift", "mean"),
        median_uplift=("forecast_revenue_uplift", "median"),
    ).sort_values("mean_uplift", ascending=True)
    selected.to_csv(SOURCE_DIR / "balanced_price_varying_windows.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(SOURCE_DIR / "uplift_by_tariff_transition.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    ax_a, ax_b = axes
    sns.histplot(100 * selected["forecast_revenue_uplift"], bins=24, color=BLUE, alpha=0.80, edgecolor="white", ax=ax_a)
    point = 100 * float(selected["forecast_revenue_uplift"].mean())
    lower = 100 * float(acceptance["bootstrap_price_varying_uplift"]["ci95_lower"])
    upper = 100 * float(acceptance["bootstrap_price_varying_uplift"]["ci95_upper"])
    ax_a.axvline(point, color=RED, lw=1.5, label=f"Mean {point:.2f}%")
    ax_a.axvspan(lower, upper, color=RED, alpha=0.12, label=f"95% bootstrap CI {lower:.2f}–{upper:.2f}%")
    ax_a.set_xlabel("Revenue uplift per price-changing window (%)")
    ax_a.set_ylabel("Window count")
    ax_a.set_title("Uplift distribution remains strictly positive")
    ax_a.legend(loc="upper right")
    panel_label(ax_a, "a")

    y = np.arange(len(transitions))
    ax_b.barh(y, 100 * transitions["mean_uplift"], color=TEAL)
    ax_b.set_yticks(y, [f"{label}  (n={n})" for label, n in zip(transitions["tariff_transition"], transitions["windows"])])
    ax_b.set_xlabel("Mean revenue uplift (%)")
    ax_b.set_title("Benefit is concentrated at tariff transitions")
    ax_b.grid(axis="x", color=LIGHT, lw=0.6)
    panel_label(ax_b, "b")
    return export_figure(fig, "optimization_uplift_distribution")


def main() -> None:
    logger = setup_logger()
    configure_style()
    input_paths = {
        "schedules": BACKTEST_DIR / "optimized_schedules.csv",
        "windows": BACKTEST_DIR / "rolling_window_metrics.csv",
        "scenarios": BACKTEST_DIR / "scenario_summary.csv",
        "representative": VALIDATION_DIR / "representative_balanced_schedule.csv",
        "by_fold": VALIDATION_DIR / "selected_scenario_by_fold.csv",
        "bootstrap": VALIDATION_DIR / "price_varying_uplift_bootstrap.csv",
        "acceptance": VALIDATION_DIR / "selected_scenario_and_acceptance.json",
    }
    for path in input_paths.values():
        if "评分所用测试集" in str(path):
            raise RuntimeError("Forbidden external scoring dataset path")
    schedules = pd.read_csv(input_paths["schedules"], parse_dates=["forecast_origin", "target_datetime"])
    windows = pd.read_csv(input_paths["windows"], parse_dates=["forecast_origin"])
    scenarios = pd.read_csv(input_paths["scenarios"])
    representative = pd.read_csv(input_paths["representative"], parse_dates=["forecast_origin", "target_datetime"])
    by_fold = pd.read_csv(input_paths["by_fold"])
    bootstrap = pd.read_csv(input_paths["bootstrap"])
    acceptance = json.loads(input_paths["acceptance"].read_text(encoding="utf-8"))

    paths = make_dispatch_overview(representative, scenarios, by_fold)
    paths.extend(make_uplift_distribution(windows, schedules, bootstrap, acceptance))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training-only chronological OOF dispatch backtest",
        "external_scoring_data_accessed": False,
        "backend": "Python/matplotlib-seaborn only",
        "figure_contract": {
            "core_conclusion": "Balanced dispatch shifts forecast-constrained generation toward higher tariffs with positive cross-fold uplift, exact conservation, and zero official hard-constraint violations.",
            "archetype": "quantitative grid",
            "exports": ["editable SVG", "PDF", "300 dpi PNG"],
        },
        "figures": paths,
        "representative_selection": acceptance["representative_window"],
        "acceptance": acceptance["acceptance"],
        "qa": {
            "source_data_exported": True,
            "editable_svg_text": True,
            "visual_encoding": "baseline/optimized use both distinct color and line treatment",
            "review_risk": "economic uplift is forecast-relative and applies to price-changing two-hour windows; it is not a causal plant intervention estimate",
        },
    }
    (OUT_DIR / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Created %d optimization figure files", len(paths))
    logger.info("Wrote optimization visualizations to %s", OUT_DIR)


if __name__ == "__main__":
    main()
