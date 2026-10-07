"""Figure B: physical feature architecture, operator composition, and provenance."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

from .export import export_figure
from .features import (
    OPERATOR_ORDER,
    SOURCE_ORDER,
    availability_counts,
    operator_counts,
    source_operator_table,
    target_feature_count,
)
from .io import Paths, feature_dictionary
from .style import PALETTE, apply_publication_style, panel_label
from .text import LABELS

_OPERATOR_COLORS = {
    "state snapshot": PALETTE["blue"],
    "rolling statistics": PALETTE["teal"],
    "dynamics": PALETTE["phase7"],
    "coverage QC": PALETTE["phase6_soft"],
    "known future": PALETTE["gold"],
}


def _draw_heatmap(ax, table, fig) -> None:
    """Draw the physical-domain × transform-family feature-count matrix."""
    panel_label(ax, "a")
    cmap = LinearSegmentedColormap.from_list(
        "feature_blue", ["#F7F9FC", "#CAD7EA", PALETTE["blue"]]
    )
    image = ax.imshow(table.values, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, table.values.max()))
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xticks(range(len(table.columns)), [v.replace(" ", "\n") for v in table.columns])
    ax.tick_params(length=0, pad=3)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    threshold = table.values.max() * 0.55
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = int(table.iat[row, col])
            if value:
                ax.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if value >= threshold else PALETTE["black"],
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.015)
    colorbar.set_label(LABELS["figureB"]["heatmap_colorbar"])
    colorbar.outline.set_visible(False)


def _draw_donut(ax, counts) -> None:
    """Draw a directly labelled transform-family composition donut."""
    panel_label(ax, "b")
    names = [name for name in OPERATOR_ORDER if counts.get(name, 0)]
    values = [counts[name] for name in names]
    colors = [_OPERATOR_COLORS[name] for name in names]
    total = sum(values)
    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.32, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.text(0, 0, LABELS["figureB"]["donut_center"], ha="center", va="center", fontsize=7.2, fontweight="bold")
    label_positions = {
        "state snapshot": (1.12, 0.42, "left"),
        "rolling statistics": (-0.78, -0.90, "center"),
        "dynamics": (-1.15, 0.38, "right"),
        "coverage QC": (-0.80, 0.96, "center"),
        "known future": (0.10, 1.17, "center"),
    }
    for name, value in zip(names, values):
        x, y, align = label_positions[name]
        label = f"{name}\n{value} · {100 * value / total:.1f}%"
        ax.text(x, y, label, ha=align, va="center", fontsize=5.4, color=PALETTE["black"])
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_waffle(ax, dictionary) -> None:
    """Draw all 276 features as tiles with direct provenance counts."""
    panel_label(ax, "c")
    counts = availability_counts(dictionary)
    causal = int(counts.get("<=origin", 0))
    known = int(counts.get("known_before_origin", 0))
    target = target_feature_count(dictionary)
    total = len(dictionary)
    cols = 18
    rows = math.ceil(total / cols)
    colors = [PALETTE["blue2"]] * causal + [PALETTE["phase7"]] * known
    colors += [PALETTE["red"]] * max(0, target)
    colors += [PALETTE["neutral_light"]] * max(0, total - len(colors))
    for index in range(total):
        row = rows - 1 - index // cols
        col = index % cols
        ax.add_patch(
            Rectangle(
                (col, row),
                0.78,
                0.78,
                facecolor=colors[index],
                edgecolor="white",
                linewidth=0.22,
            )
        )
    ax.set_xlim(-0.2, 29.0)
    ax.set_ylim(-0.5, rows + 0.3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    x = 19.2
    items = [
        (PALETTE["blue2"], f"{causal} causal\n≤ origin"),
        (PALETTE["phase7"], f"{known} known\nbefore origin"),
        (PALETTE["red"], f"{target} target-derived"),
    ]
    y_values = [12.5, 8.0, 3.7]
    for (color, label), y in zip(items, y_values):
        ax.add_patch(Rectangle((x, y), 1.05, 1.05, facecolor=color, edgecolor="none"))
        ax.text(x + 1.45, y + 0.53, label, va="center", ha="left", fontsize=5.6, color=PALETTE["black"])


def build(paths: Paths, out_dir):
    """Build Figure B from the archived 276-feature dictionary."""
    dictionary = feature_dictionary(paths)
    if dictionary is None or dictionary.empty:
        return None
    apply_publication_style()
    table = source_operator_table(dictionary).reindex(index=SOURCE_ORDER, columns=OPERATOR_ORDER)
    counts = operator_counts(dictionary)

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.05, 1.0], height_ratios=[1.0, 1.0])
    heatmap_ax = fig.add_subplot(grid[:, :2])
    donut_ax = fig.add_subplot(grid[0, 2])
    waffle_ax = fig.add_subplot(grid[1, 2])

    _draw_heatmap(heatmap_ax, table, fig)
    _draw_donut(donut_ax, counts)
    _draw_waffle(waffle_ax, dictionary)
    return export_figure(fig, out_dir, "figureB_feature_architecture")
