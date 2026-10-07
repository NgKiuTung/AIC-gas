"""Figure E: feature completeness, raw-field missingness, and anomaly isolation evidence."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .export import export_figure
from .io import Paths, changes, coverage, inventory, numeric_profile, source_audit
from .style import PALETTE, apply_publication_style, clean_axis, panel_label
from .text import LABELS


def _availability(df: pd.DataFrame, column: str) -> np.ndarray:
    """Convert archived missing fractions to bounded availability ratios."""
    values = 1.0 - pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(float)
    return np.clip(values, 0.0, 1.0)


def _jitter(count: int, amplitude: float = 0.13) -> np.ndarray:
    """Return deterministic symmetric jitter without introducing random state."""
    if count <= 1:
        return np.zeros(count)
    pattern = np.linspace(-amplitude, amplitude, 17)
    return np.resize(pattern, count)


def _draw_availability(ax, cov: pd.DataFrame) -> None:
    """Draw train/test feature availability as a compact raincloud-like strip."""
    panel_label(ax, "a")
    specs = [
        ("train_missing_fraction", 1.0, PALETTE["blue"], LABELS["figureE"]["availability_train"]),
        ("test_missing_fraction", 0.0, PALETTE["teal"], LABELS["figureE"]["availability_test"]),
    ]
    for column, base, color, label in specs:
        if column not in cov.columns:
            continue
        values = _availability(cov, column)
        ax.scatter(values, base + _jitter(len(values)), s=8, color=color, alpha=0.20, edgecolors="none")
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([q1, q3], [base, base], color=color, lw=4.5, solid_capstyle="round", zorder=4)
        ax.scatter([median], [base], marker="D", s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
        low = int(np.sum(values < 0.95))
        ax.text(0.02, base + 0.25, f"{label}: {low} / {len(values)} below 95%", fontsize=5.6, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([1.0, 0.0], [LABELS["figureE"]["availability_train"], LABELS["figureE"]["availability_test"]])
    ax.set_xlabel(LABELS["figureE"]["availability_xlabel"])
    clean_axis(ax)


def _raw_missingness(profile: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate missing counts by raw field using source-specific row denominators."""
    if not {"field", "missing", "source"}.issubset(profile.columns):
        return pd.DataFrame()
    if not {"file", "rows"}.issubset(inv.columns):
        return pd.DataFrame()
    row_lookup = inv.set_index("file")["rows"].to_dict()
    frame = profile[["field", "missing", "source"]].copy()
    frame["rows"] = frame["source"].map(row_lookup)
    frame["missing"] = pd.to_numeric(frame["missing"], errors="coerce")
    frame["rows"] = pd.to_numeric(frame["rows"], errors="coerce")
    frame = frame.dropna(subset=["missing", "rows"])
    grouped = frame.groupby("field", as_index=False).agg(missing=("missing", "sum"), rows=("rows", "sum"))
    grouped["missing_pct"] = 100.0 * grouped["missing"] / grouped["rows"].replace(0, np.nan)
    return grouped.dropna(subset=["missing_pct"])


def _draw_missingness(ax, profile: pd.DataFrame, inv: pd.DataFrame) -> None:
    """Draw the most incomplete raw process fields as percentages, not counts."""
    panel_label(ax, "b")
    data = _raw_missingness(profile, inv).nlargest(7, "missing_pct").sort_values("missing_pct")
    y = np.arange(len(data))
    ax.hlines(y, 0, data["missing_pct"], color=PALETTE["neutral_light"], lw=1.2)
    ax.scatter(data["missing_pct"], y, color=PALETTE["red"], s=28, zorder=3)
    for yi, value in zip(y, data["missing_pct"]):
        ax.text(min(value + 1.5, 99.0), yi, f"{value:.1f}%", va="center", fontsize=5.5)
    ax.set_yticks(y, data["field"].astype(str))
    ax.set_xlim(0, 104)
    ax.set_xlabel(LABELS["figureE"]["missing_xlabel"])
    clean_axis(ax)


def _draw_audit(ax, audit: pd.DataFrame, source: dict) -> None:
    """Show how anomaly-isolation events concentrate by field on a log scale."""
    panel_label(ax, "c")
    if "field" not in audit.columns:
        ax.set_axis_off()
        return
    counts = audit["field"].astype(str).value_counts().head(6).sort_values()
    y = np.arange(len(counts))
    ax.hlines(y, 0.8, counts.values, color=PALETTE["neutral_light"], lw=1.2)
    ax.scatter(counts.values, y, color=PALETTE["teal"], s=30, zorder=3)
    for yi, value in zip(y, counts.values):
        ax.text(value * 1.08, yi, f"{int(value):,}", va="center", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_yticks(y, counts.index)
    ax.set_xlabel(LABELS["figureE"]["audit_xlabel"])
    clean_axis(ax)



def build(paths: Paths, out_dir):
    """Build Figure E from archived coverage and audit metadata."""
    cov = coverage(paths)
    profile = numeric_profile(paths)
    inv = inventory(paths)
    audit = changes(paths)
    if cov is None or profile is None or inv is None or audit is None:
        return None
    apply_publication_style()
    fig = plt.figure(figsize=(7.2, 3.55), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05])
    _draw_availability(fig.add_subplot(grid[0, 0]), cov)
    _draw_missingness(fig.add_subplot(grid[0, 1]), profile, inv)
    _draw_audit(fig.add_subplot(grid[0, 2]), audit, source_audit(paths))
    return export_figure(fig, out_dir, "figureE_data_quality")
