"""Figure A: final submission provenance with parallel Long and frozen Short lanes."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .export import export_figure
from .io import Paths, parse_online_metrics, ready
from .style import PALETTE, apply_publication_style, clean_axis, panel_label
from .text import LABELS


def _box(ax, xy, wh, text, color, *, sub=None, weight="bold") -> None:
    """Draw one rounded schematic node in axes coordinates."""
    x, y = xy
    width, height = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.61,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight=weight,
        color=PALETTE["black"],
    )
    if sub:
        ax.text(
            x + width / 2,
            y + height * 0.28,
            sub,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=5.8,
            color=PALETTE["neutral"],
        )


def _arrow(ax, start, end, color=None) -> None:
    """Draw a restrained directional arrow."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.95,
            color=color or PALETTE["neutral"],
            connectionstyle="arc3,rad=0",
        )
    )


def _hero(ax, info: dict, metrics: dict) -> None:
    """Draw the two-lane system provenance without implying Short depends on Long."""
    panel_label(ax, "a")
    ax.set_axis_off()
    labels = LABELS["figureA"]
    _box(
        ax,
        (0.03, 0.60),
        (0.16, 0.22),
        labels["processed"],
        "#D9E8F5",
        sub=labels["processed_sub"].format(origins=info.get("origins", "n/a")),
    )
    _box(ax, (0.25, 0.60), (0.16, 0.22), labels["features"], "#C8D9EE", sub=labels["features_sub"])
    _box(ax, (0.47, 0.60), (0.18, 0.22), labels["long"], PALETTE["phase7_soft"], sub=labels["long_sub"])

    _box(ax, (0.25, 0.17), (0.16, 0.20), "Phase6 lineage", "#EFF1F7", sub="archived Short model", weight="normal")
    _box(ax, (0.47, 0.17), (0.18, 0.20), labels["short"], PALETTE["phase6_soft"], sub=labels["short_sub"])

    _box(
        ax,
        (0.79, 0.36),
        (0.18, 0.24),
        labels["final"],
        "#E8E8E8",
        sub=labels["final_sub"].format(score=metrics.get("total", 60.0903)),
    )

    _arrow(ax, (0.19, 0.71), (0.25, 0.71))
    _arrow(ax, (0.41, 0.71), (0.47, 0.71))
    _arrow(ax, (0.41, 0.27), (0.47, 0.27))
    _arrow(ax, (0.65, 0.71), (0.79, 0.53), PALETTE["phase7"])
    _arrow(ax, (0.65, 0.27), (0.79, 0.43), PALETTE["phase6"])

    ax.text(0.03, 0.89, "Long research lane", transform=ax.transAxes, fontsize=6.5, fontweight="bold", color=PALETTE["phase7"])
    ax.text(0.03, 0.30, "Frozen Short lane", transform=ax.transAxes, fontsize=6.5, fontweight="bold", color=PALETTE["phase6"])


def _scale_panel(ax, info: dict) -> None:
    """Show data/feature scale as dot-and-stem counts on a log axis."""
    panel_label(ax, "b")
    labels = LABELS["figureA"]
    values = [info.get("origins"), info.get("test_origins"), info.get("feature_count")]
    names = [labels["scale_all"], labels["scale_test"], labels["scale_features"]]
    colors = [PALETTE["blue2"], PALETTE["teal"], PALETTE["phase7"]]
    for row, (value, color) in enumerate(zip(values, colors)):
        if value is None:
            continue
        value = float(value)
        ax.hlines(row, 1, value, color=PALETTE["neutral_light"], lw=1.1)
        ax.scatter([value], [row], s=30, color=color, zorder=3)
        ax.text(value, row + 0.13, f"{value:,.0f}", fontsize=6.4, ha="center", va="bottom")
    ax.set_yticks(range(len(names)), names)
    ax.set_xscale("log")
    ax.set_xlabel(labels["scale_xlabel"])
    ax.invert_yaxis()
    clean_axis(ax)


def _contribution_panel(ax, metrics: dict) -> None:
    """Show the two frozen 50-point online contributions."""
    panel_label(ax, "c")
    labels = LABELS["figureA"]
    rows = [
        (LABELS["common"]["short"], metrics.get("short_score"), PALETTE["phase6"]),
        (LABELS["common"]["long"], metrics.get("long_score"), PALETTE["phase7"]),
    ]
    for row, (name, value, color) in enumerate(rows):
        if value is None:
            continue
        ax.hlines(row, 0, 50, color=PALETTE["neutral_light"], lw=1.7)
        ax.hlines(row, 0, value, color=color, lw=3.3)
        ax.scatter([value], [row], color=color, s=30, zorder=4)
        ax.text(value + 0.8, row, f"{value:.4f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 50)
    ax.set_yticks([0, 1], [LABELS["common"]["short"], LABELS["common"]["long"]])
    ax.set_xlabel(labels["contrib_xlabel"])
    ax.invert_yaxis()
    clean_axis(ax)


def build(paths: Paths, out_dir):
    """Build Figure A from frozen Phase7.1 metadata and online metrics."""
    apply_publication_style()
    info = ready(paths)
    metrics = parse_online_metrics(paths)
    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.75, 1.0], width_ratios=[1.35, 1.0])
    _hero(fig.add_subplot(grid[0, :]), info, metrics)
    _scale_panel(fig.add_subplot(grid[1, 0]), info)
    _contribution_panel(fig.add_subplot(grid[1, 1]), metrics)
    return export_figure(fig, out_dir, "figureA_pipeline_and_score")
