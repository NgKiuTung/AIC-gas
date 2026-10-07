"""Figure D: online progression, score contribution, and accuracy comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .export import export_figure
from .io import Paths, parse_online_metrics
from .style import PALETTE, apply_publication_style, clean_axis, panel_label
from .text import LABELS


def _progression(ax, total: float) -> None:
    """Draw discrete online submission states with explicit score deltas."""
    panel_label(ax, "a")
    labels = LABELS["figureD"]
    stages = [labels["stage_reference"], labels["stage_probe"], labels["stage_final"]]
    values = [58.1578, 58.56, total]
    colors = [PALETTE["neutral"], PALETTE["teal"], PALETTE["phase7"]]
    x = np.arange(3)
    ax.plot(x, values, color=PALETTE["neutral_light"], lw=1.5, zorder=1)
    for xi, yi, color in zip(x, values, colors):
        ax.scatter([xi], [yi], s=46, color=color, zorder=3)
        ax.text(xi, yi + 0.14, f"{yi:.4f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    for left in range(2):
        delta = values[left + 1] - values[left]
        ax.text(left + 0.5, (values[left] + values[left + 1]) / 2 + 0.11, f"+{delta:.4f}", ha="center", fontsize=5.8, color=PALETTE["ink_soft"])
    ax.set_xticks(x, stages)
    ax.set_ylim(min(values) - 0.45, max(values) + 0.55)
    ax.set_ylabel(labels["score_ylabel"])
    clean_axis(ax)


def _dumbbell(ax, first: float, second: float, *, xlabel: str, first_label: str, second_label: str, suffix: str, decimals: int) -> None:
    """Draw a compact two-point comparison with semantic Phase6/Phase7 colors."""
    ax.plot([second, first], [0, 0], color=PALETTE["neutral_light"], lw=2.2)
    ax.scatter([first], [0], s=48, color=PALETTE["phase6"], zorder=3)
    ax.scatter([second], [0], s=48, color=PALETTE["phase7"], zorder=3)
    first_text = f"{first_label} {first:.{decimals}f}{suffix}"
    second_text = f"{second_label} {second:.{decimals}f}{suffix}"
    ax.text(first, 0.12, first_text, ha="center", va="bottom", fontsize=6.5, color=PALETTE["phase6"])
    ax.text(second, -0.12, second_text, ha="center", va="top", fontsize=6.5, color=PALETTE["phase7"])
    midpoint = (first + second) / 2
    delta = abs(first - second)
    unit = " pp" if suffix == "%" else ""
    ax.text(midpoint, 0.0, f"Δ {delta:.{decimals}f}{unit}", ha="center", va="bottom", fontsize=5.2, color=PALETTE["ink_soft"])
    pad = max(0.7, delta * 0.25)
    ax.set_xlim(min(first, second) - pad, max(first, second) + pad)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    clean_axis(ax, keep_left=False)


def build(paths: Paths, out_dir):
    """Build Figure D from frozen online metrics."""
    metrics = parse_online_metrics(paths)
    if not metrics:
        return None
    apply_publication_style()
    fig = plt.figure(figsize=(7.2, 3.05), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.95])

    _progression(fig.add_subplot(grid[0, 0]), metrics.get("total", 60.0903))

    ax1 = fig.add_subplot(grid[0, 1])
    panel_label(ax1, "b")
    if metrics.get("short_score") is not None and metrics.get("long_score") is not None:
        _dumbbell(
            ax1,
            metrics["short_score"],
            metrics["long_score"],
            xlabel=LABELS["figureD"]["contribution_xlabel"],
            first_label=LABELS["common"]["short"],
            second_label=LABELS["common"]["long"],
            suffix="",
            decimals=4,
        )

    ax2 = fig.add_subplot(grid[0, 2])
    panel_label(ax2, "c")
    if metrics.get("short_accuracy") is not None and metrics.get("long_accuracy") is not None:
        _dumbbell(
            ax2,
            metrics["short_accuracy"],
            metrics["long_accuracy"],
            xlabel=LABELS["figureD"]["accuracy_xlabel"],
            first_label=LABELS["common"]["short"],
            second_label=LABELS["common"]["long"],
            suffix="%",
            decimals=2,
        )

    return export_figure(fig, out_dir, "figureD_online_performance")
