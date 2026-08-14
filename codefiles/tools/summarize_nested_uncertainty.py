"""Paired outer-window bootstrap for descriptive nested-validation uncertainty."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "experiments" / "nested_temporal_validation" / "outer_window_summary.csv"
OUTPUT_PATH = ROOT / "results" / "experiments" / "nested_temporal_validation" / "uncertainty_summary.json"
SEED = 20260814
BOOTSTRAP_DRAWS = 20_000


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def main() -> None:
    windows = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    model = windows["model_mape"].to_numpy(dtype=np.float64)
    persistence = windows["persistence_mape"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(windows), size=(BOOTSTRAP_DRAWS, len(windows)))
    model_draws = model[samples].mean(axis=1)
    persistence_draws = persistence[samples].mean(axis=1)
    improvement_draws = persistence_draws - model_draws
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paired outer-window nonparametric bootstrap",
        "descriptive_only_not_used_for_model_selection": True,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "outer_windows": len(windows),
        "model_mape_quantiles_2.5_50_97.5": interval(model_draws),
        "persistence_mape_quantiles_2.5_50_97.5": interval(persistence_draws),
        "absolute_improvement_quantiles_2.5_50_97.5": interval(improvement_draws),
        "bootstrap_probability_improvement": float(np.mean(improvement_draws > 0)),
        "observed_windows_improved": int(np.sum(model < persistence)),
        "observed_windows_total": len(windows),
        "external_scoring_data_accessed": False,
        "caveat": (
            "Five outer windows give limited uncertainty resolution; "
            "historical architecture selection remains post hoc."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
