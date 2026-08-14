"""Final regime and temporal error tables for the selected hybrid OOF model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_PATH = ROOT / "results" / "experiments" / "hybrid_target" / "hybrid_oof_predictions.csv"
FEATURE_PATH = ROOT / "results" / "features" / "train_supervised_features.pkl"
RESULT_DIR = ROOT / "results" / "experiments" / "final_hybrid_diagnostics"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(PREDICTION_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    features = pd.read_pickle(FEATURE_PATH)
    features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
    rows = rows.merge(
        features[["datetime", "feat_diff4_p50_current", "feat_diff4_generator_all_filled"]],
        on="datetime", how="left", validate="many_to_one",
    )
    diff = np.where(
        rows["target"].eq("generator_1"),
        np.abs(rows["feat_diff4_p50_current"]),
        np.abs(rows["feat_diff4_generator_all_filled"]),
    )
    rows["observed_change_ratio_1h"] = diff / np.maximum(np.abs(rows["current"]), 1e-6)
    rows["operating_regime"] = ""
    for target in ("generator_1", "generator_all"):
        mask = rows["target"].eq(target)
        unique_origins = rows.loc[mask].drop_duplicates("datetime")["observed_change_ratio_1h"]
        q50, q80 = unique_origins.quantile([0.5, 0.8]).to_numpy()
        rows.loc[mask, "operating_regime"] = pd.cut(
            rows.loc[mask, "observed_change_ratio_1h"],
            [-np.inf, q50, q80, np.inf],
            labels=["stable_0_50pct", "changing_50_80pct", "volatile_80_100pct"],
        ).astype(str)
    forecast_time = rows["datetime"] + pd.to_timedelta(rows["horizon_minutes"], unit="m")
    hour = forecast_time.dt.hour
    rows["forecast_session"] = pd.cut(
        hour, [-1, 5, 8, 16, 21, 23],
        labels=["night_00_06", "morning_06_09", "day_09_17", "evening_17_22", "late_22_24"],
    ).astype(str)
    rows["weekday_weekend"] = np.where(forecast_time.dt.dayofweek >= 5, "weekend", "weekday")
    rows["improvement"] = rows["ape_persistence"] - rows["ape_hybrid"]

    regime = rows.groupby(["target", "operating_regime"], as_index=False).agg(
        samples=("actual", "size"), persistence_mape=("ape_persistence", "mean"),
        hybrid_mape=("ape_hybrid", "mean"), absolute_mape_reduction=("improvement", "mean")
    )
    regime["relative_mape_reduction"] = regime["absolute_mape_reduction"] / regime["persistence_mape"]
    temporal = rows.groupby(["target", "forecast_session"], as_index=False).agg(
        samples=("actual", "size"), persistence_mape=("ape_persistence", "mean"),
        hybrid_mape=("ape_hybrid", "mean"), absolute_mape_reduction=("improvement", "mean")
    )
    weekpart = rows.groupby(["target", "weekday_weekend"], as_index=False).agg(
        samples=("actual", "size"), persistence_mape=("ape_persistence", "mean"),
        hybrid_mape=("ape_hybrid", "mean"), absolute_mape_reduction=("improvement", "mean")
    )
    horizon = rows.groupby(["target", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), hybrid_mape=("ape_hybrid", "mean"),
        absolute_mape_reduction=("improvement", "mean")
    )
    target = rows.groupby("target", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), hybrid_mape=("ape_hybrid", "mean"),
        absolute_mape_reduction=("improvement", "mean")
    )
    regime.to_csv(RESULT_DIR / "metrics_by_operating_regime.csv", index=False, encoding="utf-8-sig")
    temporal.to_csv(RESULT_DIR / "metrics_by_forecast_session.csv", index=False, encoding="utf-8-sig")
    weekpart.to_csv(RESULT_DIR / "metrics_by_weekpart.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(RESULT_DIR / "metrics_by_target_horizon.csv", index=False, encoding="utf-8-sig")
    target.to_csv(RESULT_DIR / "metrics_by_target.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "selected training-only out-of-fold hybrid predictions",
        "external_scoring_data_accessed": False,
        "rows": len(rows),
        "operating_regime_rule": "target-specific quantiles of observed one-hour relative change at forecast origin",
        "all_regimes_improve": bool((regime["absolute_mape_reduction"] > 0).all()),
        "all_sessions_improve": bool((temporal["absolute_mape_reduction"] > 0).all()),
        "target_metrics": target.to_dict(orient="records"),
    }
    (RESULT_DIR / "diagnostic_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
