"""Figure C: screening, confirmation stability, and final component selection."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .export import export_figure
from .io import Paths, confirmation, screening, selection
from .style import PALETTE, apply_publication_style, clean_axis, panel_label
from .text import LABELS

BANDS = ["h01_08", "h09_24", "h25_48", "h49_96"]

_CANDIDATE_LABELS = {
    "xgb_wide_180d_d50_t12": "XGB 180d t12",
    "xgb_wide_180d_d50_t9": "XGB 180d t9",
    "xgb_wide_p6_180d_d50_t6": "XGB p6 t6",
    "xgb_wide_90d_d50_t6": "XGB 90d t6",
    "xgb_wide_120d_d50_t6": "XGB 120d t6",
    "xgb_wide_180d_d50_t6_l31": "XGB 180d t6 L31",
    "tcn_long_residual": "TCN residual",
    "xgb_wide_180d_d25_t6": "XGB 180d d25 t6",
    "reference_58": "reference",
}


def _short(name: str) -> str:
    """Return a stable display name without mutating canonical model casing."""
    return _CANDIDATE_LABELS.get(str(name), str(name))


def _screen_frame(obj: dict) -> pd.DataFrame:
    """Flatten screening component gates."""
    rows: list[dict] = []
    for candidate, payload in obj.items():
        components = payload.get("components", {}) if isinstance(payload, dict) else {}
        for component, data in components.items():
            if not isinstance(data, dict):
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "component": component,
                    "gain": data.get("mean_gain"),
                    "accepted": bool(data.get("accepted", False)),
                }
            )
    return pd.DataFrame(rows)


def _confirm_frame(obj: dict) -> pd.DataFrame:
    """Flatten confirmation diagnostics."""
    rows: list[dict] = []
    for component, data in obj.get("components", {}).items():
        if not isinstance(data, dict):
            continue
        rows.append(
            {
                "component": component,
                "candidate": data.get("candidate", "reference_58"),
                "mean_gain": data.get("mean_gain"),
                "worst_gain": data.get("worst_gain"),
                "last_gain": data.get("last_gain"),
                "accepted": bool(data.get("accepted", False)),
            }
        )
    return pd.DataFrame(rows)


def _draw_screening(ax, screen: pd.DataFrame, fig) -> None:
    """Draw the screening gain matrix and highlight the winning candidate row."""
    panel_label(ax, "a")
    screen = screen.dropna(subset=["gain"]).copy()
    components = [
        component
        for component in [f"g1__{band}" for band in BANDS] + [f"gall__{band}" for band in BANDS]
        if component in set(screen["component"])
    ]
    ranking = screen.groupby("candidate")["gain"].mean().sort_values(ascending=False)
    candidates = ranking.head(8).index.tolist()
    pivot = screen.pivot_table(index="candidate", columns="component", values="gain", aggfunc="first")
    pivot = pivot.reindex(index=candidates, columns=components)
    vmax = max(0.001, float(np.nanmax(np.abs(pivot.values))))
    cmap = LinearSegmentedColormap.from_list(
        "gain", [PALETTE["red_soft"], "#FAFAFA", PALETTE["green_soft"]]
    )
    image = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
    )
    ax.set_yticks(range(len(pivot.index)), [_short(value) for value in pivot.index])
    ax.set_xticks(
        range(len(pivot.columns)),
        [value.split("__", 1)[1].replace("_", "–") for value in pivot.columns],
    )
    ax.tick_params(length=0, pad=2)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    if "xgb_wide_180d_d50_t12" in pivot.index:
        row = pivot.index.get_loc("xgb_wide_180d_d50_t12")
        ax.add_patch(
            Rectangle(
                (-0.49, row - 0.48),
                len(pivot.columns) - 0.02,
                0.96,
                fill=False,
                edgecolor=PALETTE["phase7"],
                linewidth=1.2,
            )
        )
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=5.3)
    if len(components) >= 8:
        ax.axvline(3.5, color="white", lw=2.0)
        ax.text(1.5, -0.88, LABELS["figureC"]["screen_group_g1"], ha="center", va="bottom", fontsize=6.0, fontweight="bold")
        ax.text(5.5, -0.88, LABELS["figureC"]["screen_group_gall"], ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label(LABELS["figureC"]["gain_colorbar"])
    colorbar.outline.set_visible(False)


def _draw_confirmation(ax, confirm: pd.DataFrame) -> None:
    """Draw accepted generator_all confirmation diagnostics around the zero-gain line."""
    panel_label(ax, "b")
    data = confirm[
        confirm["component"].astype(str).str.startswith("gall__")
        & confirm["mean_gain"].notna()
    ].copy()
    order = [f"gall__{band}" for band in BANDS]
    data["order"] = data["component"].map({name: index for index, name in enumerate(order)})
    data = data.sort_values("order")
    y = np.arange(len(data))[::-1]
    for yi, (_, row) in zip(y, data.iterrows()):
        mean = float(row["mean_gain"])
        worst = float(row["worst_gain"]) if pd.notna(row["worst_gain"]) else mean
        last = float(row["last_gain"]) if pd.notna(row["last_gain"]) else mean
        lo, hi = min(worst, last), max(worst, last)
        color = PALETTE["green"] if bool(row["accepted"]) else PALETTE["red"]
        ax.plot([lo, hi], [yi, yi], color=PALETTE["neutral"], lw=1.3)
        ax.plot([worst, worst], [yi - 0.10, yi + 0.10], color=PALETTE["neutral"], lw=0.9)
        ax.scatter([mean], [yi], color=color, s=28, zorder=3)
        ax.scatter([last], [yi], facecolor="white", edgecolor=color, marker="D", s=20, zorder=4)
    ax.axvline(0, color=PALETTE["neutral_light"], lw=0.9)
    ax.set_yticks(y, [band.replace("_", "–") for band in BANDS])
    ax.set_xlabel(LABELS["figureC"]["confirm_xlabel"])
    clean_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["green"], markeredgecolor=PALETTE["green"], label="mean"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=PALETTE["green"], label="last fold"),
        Line2D([0], [0], color=PALETTE["neutral"], lw=1.2, label="worst↔last"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=4.8, handlelength=1.0, columnspacing=0.8)


def _selection_map(obj: dict) -> dict[tuple[int, int], str]:
    """Resolve final picks from screen_picks, preserving reference components."""
    long = obj.get("long", {}) if isinstance(obj, dict) else {}
    picks = long.get("screen_picks", {}) if isinstance(long, dict) else {}
    if not picks:
        picks = {f"g1__{band}": "reference_58" for band in BANDS}
        accepted = long.get("accepted_components", {}) if isinstance(long, dict) else {}
        picks.update(accepted)
    mapping: dict[tuple[int, int], str] = {}
    for row, prefix in enumerate(("g1", "gall")):
        for col, band in enumerate(BANDS):
            mapping[(row, col)] = str(picks.get(f"{prefix}__{band}", "reference_58"))
    return mapping


def _draw_selection(ax, obj: dict) -> None:
    """Draw the final horizon selection strip from authoritative screen_picks."""
    panel_label(ax, "c")
    mapping = _selection_map(obj)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.set_xticks(np.arange(4) + 0.5, [band.replace("_", "–") for band in BANDS])
    ax.set_yticks([1.5, 0.5], [LABELS["figureC"]["generator_1"], LABELS["figureC"]["generator_all"]])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    for row in range(2):
        for col in range(4):
            candidate = mapping[(row, col)]
            is_reference = candidate == "reference_58"
            color = PALETTE["neutral_light"] if is_reference else PALETTE["phase7_soft"]
            text_color = PALETTE["black"]
            ax.add_patch(Rectangle((col + 0.03, (1 - row) + 0.09), 0.94, 0.82, facecolor=color, edgecolor="white", linewidth=1.0))
            ax.text(col + 0.50, (1 - row) + 0.50, _short(candidate), ha="center", va="center", fontsize=6.2, color=text_color, fontweight="bold" if not is_reference else "normal")


def build(paths: Paths, out_dir):
    """Build Figure C from frozen screening, confirmation, and selection metadata."""
    apply_publication_style()
    screen = _screen_frame(screening(paths))
    confirm = _confirm_frame(confirmation(paths))
    selected = selection(paths)

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.55, 0.78], width_ratios=[1.25, 1.05, 0.85])
    heatmap_ax = fig.add_subplot(grid[0, :2])
    confirm_ax = fig.add_subplot(grid[0, 2])
    select_ax = fig.add_subplot(grid[1, :])
    if not screen.empty and screen["gain"].notna().any():
        _draw_screening(heatmap_ax, screen, fig)
    else:
        heatmap_ax.set_axis_off()
    _draw_confirmation(confirm_ax, confirm)
    _draw_selection(select_ax, selected)
    return export_figure(fig, out_dir, "figureC_model_selection_evidence")
