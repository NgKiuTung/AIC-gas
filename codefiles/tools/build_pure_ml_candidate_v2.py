"""Train-only GPU LightGBM candidate with replayable raw-named input schema.

Retrospective diagnostics only; never promotes or accesses scoring inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables
from gas_power.data.raw_inputs import load_price_lookup, load_training_raw_tables
from gas_power.features.inference import build_inference_feature_frame
from gas_power.features.spectral import add_causal_spectral_features
from gas_power.forecasting.final_inference import load_feature_schema
from gas_power.submission.package import build_submission_zip, validate_submission_zip
from gas_power.submission.schema import build_submission_frame

ROOT = Path(__file__).resolve().parents[2]
SIGNALS = {
    "p50": "feat_generator_1_filled",
    "pall": "feat_generator_all_filled",
    "holder": "blast_furnace_gas_holder_2",
    "gen_bfg": "generator_use_blast_furnace_gas",
    "gen_cog": "generator_use_coke_gas",
    "gen_cvg": "generator_use_converter_gas",
    "bfg_supply": "feat_blast_furnace_observed_sum",
    "bfg_users": "feat_blast_furnace_user_observed_sum",
}
PARAMS = dict(
    objective="regression_l1",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=30,
    feature_fraction=0.7,
    lambda_l1=1.0,
    lambda_l2=20.0,
    num_threads=4,
    verbosity=-1,
    device_type="gpu",
    seed=20260905,
)
ROUNDS = 120


def candidate_features(causal, frozen_schema):
    base = build_inference_feature_frame(causal)
    keep = [c for c in frozen_schema if "converter_balance_proxy" not in c]
    base = base[keep].copy()
    # Fill at the origin is causal and exposed by the existing missing flags.
    base.insert(0, "generator_all", causal["feat_generator_all_filled"].to_numpy())
    base.insert(0, "generator_1", causal["feat_generator_1_filled"].to_numpy())
    spectral = add_causal_spectral_features(causal, SIGNALS, window=96).drop(columns="datetime")
    result = pd.concat([base, spectral], axis=1).astype("float32")
    result.index = pd.DatetimeIndex(causal.datetime, name="datetime")
    return result


def main():
    out = ROOT / "results/experiments/pure_ml_candidate_v2" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    root = ROOT / "dataset/初赛-数据集"
    tables = load_training_raw_tables(root)
    tables = {k: v.loc[v.datetime < pd.Timestamp("2025-05-01")].copy() for k, v in tables.items()}
    causal, _ = preprocess_causal_raw_tables(tables, load_price_lookup(root / "price.xlsx"))
    schema = load_feature_schema(ROOT / "results/models/production_forecaster")
    X = candidate_features(causal, schema)
    # Numerical parity of raw-named current loads with their old feature views.
    assert np.array_equal(X.generator_1.to_numpy(), causal.feat_generator_1_filled.to_numpy(dtype="float32"))
    truth = tables["load"].set_index("datetime")
    times = X.index
    y = np.column_stack(
        [
            truth[t].reindex(times + pd.Timedelta(minutes=h)).to_numpy()
            for t in ("generator_1", "generator_all")
            for h in range(15, 121, 15)
        ]
    )
    current = np.repeat(X[["generator_1", "generator_all"]].to_numpy(), 8, axis=1)
    residual = y - current
    valid = np.isfinite(X).all(axis=1).to_numpy() & np.isfinite(y).all(axis=1)
    pd.DataFrame({"position": range(len(X.columns)), "feature": X.columns}).to_csv(
        out / "feature_schema.csv", index=False
    )
    # Prefix check includes FFT/Haar, using only already-computed causal prefix.
    cut = int(np.flatnonzero(times == pd.Timestamp("2025-04-01"))[0])
    prefix = candidate_features(causal.iloc[: cut + 1], schema)
    assert np.allclose(prefix.iloc[-1], X.iloc[cut], rtol=1e-6, atol=1e-6)
    predictions = []
    metrics = []
    for start_text in ("2025-04-01", "2025-04-15", "2025-04-22", "full_refit"):
        full = start_text == "full_refit"
        start = pd.Timestamp("2025-05-01" if full else start_text)
        fit = valid & (times + pd.Timedelta(minutes=120) < start)
        select = (
            valid & (times >= start) & (times < start + pd.Timedelta(days=2))
            if not full
            else valid & (times >= pd.Timestamp("2025-04-29"))
        )
        folder = out / start_text
        folder.mkdir()
        correction = []
        logging.info("%s GPU fit=%d rows", start_text, int(fit.sum()))
        for j in range(16):
            model = lgb.train(PARAMS, lgb.Dataset(X.loc[fit], label=residual[fit, j]), num_boost_round=ROUNDS)
            path = folder / f"target_{j:02d}.txt"
            model.save_model(str(path))
            before = model.predict(X.loc[select])
            reloaded = lgb.Booster(model_file=str(path))
            after = reloaded.predict(X.loc[select], num_threads=4)
            if not np.allclose(before, after, atol=1e-8, rtol=1e-8):
                raise AssertionError("Model reload mismatch")
            correction.append(after)
        pred = np.maximum(current[select] + np.column_stack(correction), 0)
        pred[:, 8:] = np.maximum(pred[:, 8:], pred[:, :8])
        inputs = X.loc[select].reset_index()
        result = build_submission_frame(times[select], {"generator_1": pred[:, :8], "generator_all": pred[:, 8:]})
        # Full-fit replay is only a load/serialization check, never scored as validation.
        path = folder / "diagnostic_gas_predict_prelim.zip"
        build_submission_zip(result, inputs, path)
        validate_submission_zip(
            path,
            expected_datetimes=times[select],
            expected_team_name="diagnostic",
            expected_feature_schema=list(X.columns),
        )
        if not full:
            for j in range(16):
                actual = y[select, j]
                predictions.append(
                    pd.DataFrame(
                        {
                            "origin": times[select],
                            "window": start_text,
                            "target": "generator_1" if j < 8 else "generator_all",
                            "horizon": (j % 8 + 1) * 15,
                            "actual": actual,
                            "prediction": pred[:, j],
                            "ape": np.abs(pred[:, j] - actual) / np.maximum(np.abs(actual), 1e-8),
                            "persistence_ape": np.abs(current[select, j] - actual) / np.maximum(np.abs(actual), 1e-8),
                        }
                    )
                )
            metrics.append(
                {
                    "window": start_text,
                    "origins": int(select.sum()),
                    "mape_pct": float(np.mean(np.abs(pred - y[select]) / np.maximum(np.abs(y[select]), 1e-8)) * 100),
                }
            )
    detail = pd.concat(predictions, ignore_index=True)
    detail.to_csv(out / "predictions.csv.gz", index=False)
    pd.DataFrame(metrics).to_csv(out / "window_metrics.csv", index=False)
    manifest = {
        "official_scoring_data_accessed": False,
        "scope": "training_only_retrospective_candidate",
        "promoted": False,
        "formal_submission_generated": False,
        "feature_count": X.shape[1],
        "raw_current_load_fields": ["generator_1", "generator_all"],
        "all_null_policy": "excluded_not_fabricated",
        "model_family": "LightGBM",
        "parameters": PARAMS,
        "rounds": ROUNDS,
        "residual_mode": "absolute",
        "pooled_mape_pct": float(detail.ape.mean() * 100),
        "persistence_mape_pct": float(detail.persistence_ape.mean() * 100),
        "model_reload_checks_passed": True,
        "prefix_check_passed": True,
        "package_checks_passed": True,
        "full_refit_train_max_target_time": str(times[valid].max() + pd.Timedelta(minutes=120)),
        "full_refit_diagnostic_is_in_sample_not_scored": True,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.csv"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Complete %s %s", manifest, out)


if __name__ == "__main__":
    main()
