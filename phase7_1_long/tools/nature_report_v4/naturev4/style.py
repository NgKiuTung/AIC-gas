"""Publication style derived from the public nature-skills nature-figure guidance."""
from __future__ import annotations

import matplotlib as mpl

PALETTE = {
    "phase6": "#484878",
    "phase6_soft": "#B4C0E4",
    "phase7": "#9A4D8E",
    "phase7_soft": "#E4CCD8",
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "teal": "#42949E",
    "green": "#2E9E44",
    "green_soft": "#AADCA9",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
    "black": "#272727",
    "white": "#FFFFFF",
    "gold": "#D6A84B",
    "ink_soft": "#5B5B5B",
}


def apply_publication_style() -> None:
    """Apply a grid-free Nature-style Matplotlib contract with editable text."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )


def clean_axis(ax, *, keep_left: bool = True, keep_bottom: bool = True) -> None:
    """Keep only information-bearing spines and disable all grid lines."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=3, color="#666666")


def panel_label(ax, label: str) -> None:
    """Place a small bold lowercase panel label at the top-left."""
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["black"],
        clip_on=False,
    )
