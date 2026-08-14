"""Plot aggregate diagnostics for the nested temporal pseudo-test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "results" / "experiments" / "nested_temporal_validation"
OUTPUT_DIR = ROOT / "results" / "figures_safe" / "phase1"


def main() -> None:
    summary = pd.read_csv(INPUT_DIR / "outer_window_summary.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(INPUT_DIR / "outer_metrics_by_target_horizon.csv", encoding="utf-8-sig")
    selection = pd.read_csv(INPUT_DIR / "selection_trace.csv", encoding="utf-8-sig")
    report = json.loads((INPUT_DIR / "nested_validation_summary.json").read_text(encoding="utf-8"))
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    x = np.arange(len(summary))
    width = 0.36
    axes[0, 0].bar(x - width / 2, summary["persistence_mape"] * 100, width, label="Persistence")
    axes[0, 0].bar(x + width / 2, summary["model_mape"] * 100, width, label="Nested-selected")
    axes[0, 0].set_xticks(x, summary["outer_window"])
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("A. Two-day pseudo-test performance")
    axes[0, 0].legend(frameon=False)

    colors = np.where(summary["absolute_improvement_pp"] >= 0, "#2A9D8F", "#E76F51")
    axes[0, 1].bar(summary["outer_window"], summary["absolute_improvement_pp"], color=colors)
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_ylabel("MAPE reduction (percentage points)")
    axes[0, 1].set_title("B. Improvement over persistence")

    horizon = metrics.groupby("horizon_minutes", as_index=False).agg(
        persistence_mape=("persistence_mape", "mean"), model_mape=("model_mape", "mean")
    )
    axes[1, 0].plot(horizon["horizon_minutes"], horizon["persistence_mape"] * 100, marker="o", label="Persistence")
    axes[1, 0].plot(horizon["horizon_minutes"], horizon["model_mape"] * 100, marker="o", label="Nested-selected")
    axes[1, 0].set_xlabel("Forecast horizon (minutes)")
    axes[1, 0].set_ylabel("MAPE (%)")
    axes[1, 0].set_title("C. Horizon degradation")
    axes[1, 0].legend(frameon=False)

    target_window = selection.pivot(index="outer_window", columns="target", values="outer_model_mape") * 100
    sns.heatmap(target_window.T, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "MAPE (%)"})
    axes[1, 1].set_xlabel("Outer window")
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_title("D. Target-by-window error")

    fig.suptitle(
        f"Nested temporal pseudo-test | MAPE {report['nested_outer_mape'] * 100:.3f}% | "
        f"Score {report['nested_outer_score']:.4f}",
        fontsize=18,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"nested_temporal_validation.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
