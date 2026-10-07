"""One-factor September EXP-B ablations with a selectable process proxy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
STAGE22 = ROOT / "wjt" / "gas_stage22_rebuild"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE22))
import gas22.bootstrap  # noqa: F401  # registers vendored gas2/gasstage paths
from gasrefine.periodic import template
from gasstage.baselines import reconcile
from gasstage.targets import future_truth, history_only

from features_experiments.exp_bc.baseline import (
    CUTOFF,
    END,
    TARGETS,
    load_bundle,
    score,
    score_by_horizon,
)


@dataclass
class ProxyFit:
    models: list[lgb.Booster | xgb.Booster]
    centers: list[float]
    columns: list[str]
    active: np.ndarray
    history: pd.DataFrame
    cutoff: pd.Timestamp
    training_seconds: float
    engine: str

    def current(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.columns].to_numpy(dtype="float32")[:, self.active]
        if self.engine == "xgboost":
            matrix = xgb.DMatrix(x, nthread=4)
            return np.column_stack(
                [
                    model.predict(matrix) + center
                    for model, center in zip(self.models, self.centers, strict=True)
                ]
            )
        return np.column_stack(
            [
                model.predict(x, num_threads=4) + center
                for model, center in zip(self.models, self.centers, strict=True)
            ]
        )

    @classmethod
    def load(cls, directory: Path, history: pd.DataFrame) -> ProxyFit:
        metadata = json.loads((directory / "fit.json").read_text(encoding="utf-8"))
        engine = metadata.get("engine", "lightgbm")
        if engine == "xgboost":
            models = []
            for target in range(2):
                model = xgb.Booster(params={"nthread": 4})
                model.load_model(str(directory / f"target_{target}.json"))
                models.append(model)
        elif engine == "lightgbm":
            models = [
                lgb.Booster(model_file=str(directory / f"target_{target}.txt"))
                for target in range(2)
            ]
        else:
            raise ValueError(f"Unsupported proxy engine: {engine}")
        return cls(
            models,
            metadata["centers"],
            metadata["features"],
            np.isin(np.arange(len(metadata["features"])), metadata["active"]),
            history,
            pd.Timestamp(metadata["cutoff"]),
            float(metadata.get("training_seconds", 0.0)),
            engine,
        )


def add_mechanism(frame: pd.DataFrame) -> pd.DataFrame:
    """Only same-gas-type sums; all operands are already causal current features."""
    result = frame.copy()

    def f(col: str) -> pd.Series:
        return result[f"feat_{col}__current"]

    shared_units = (1, 2, 4, 5)
    result["mechanism_bf_production_observed_4of5"] = pd.concat(
        [f(f"blast_furnace_{i}") for i in shared_units], axis=1
    ).sum(axis=1, min_count=4)
    result["mechanism_air_heater_bf_use_observed_4of5"] = pd.concat(
        [f(f"air_heater_{i}") for i in shared_units], axis=1
    ).sum(axis=1, min_count=4)
    result["mechanism_bf_user_use"] = pd.concat(
        [f(f"blast_furnace_user{i}") for i in range(1, 5)], axis=1
    ).sum(axis=1, min_count=4)
    result["mechanism_bf_consumption_observed_partial"] = (
        result["mechanism_air_heater_bf_use_observed_4of5"]
        + result["mechanism_bf_user_use"]
        + f("generator_use_blast_furnace_gas")
        + f("into_gas_mixed_blast_furnace")
    )
    result["mechanism_bf_balance_partial_proxy"] = (
        result["mechanism_bf_production_observed_4of5"]
        - result["mechanism_bf_consumption_observed_partial"]
    )
    return result


def load_feature_tables(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    pre = pd.read_parquet(data_dir / "common_features_preliminary_15m.parquet").drop(
        columns="dataset_phase"
    )
    semi = pd.read_parquet(data_dir / "common_features_semifinal_15m.parquet").drop(
        columns="dataset_phase"
    )
    training = pd.read_parquet(data_dir / "train_common_jan_sep_15m.parquet")
    label_columns = [f"proxy_current_{target}" for target in TARGETS]
    pre_labels = training.loc[
        training["dataset_phase"].eq("preliminary"), label_columns
    ]
    semi_labels = training.loc[training["dataset_phase"].eq("semifinal"), label_columns]
    # The preliminary endpoint at May 3 00:00 shares an origin with the first
    # Semi process reading; its proxy pair would otherwise mix both sources.
    pre = pre.loc[pre.index < semi.index.min()]
    pre_y = pre_labels.reindex(pre.index).to_numpy()
    semi_y = semi_labels.reindex(semi.index).to_numpy()
    return pre, semi, pre_y, semi_y


def fit_proxy(
    pre: pd.DataFrame,
    semi: pd.DataFrame,
    pre_y: np.ndarray,
    semi_y: np.ndarray,
    cutoff: pd.Timestamp,
    result_dir: Path,
    *,
    columns: list[str] | None = None,
    half_life_days: float = 0,
    lookback_days: int = 120,
    engine: str = "xgboost",
) -> ProxyFit:
    if engine not in ("lightgbm", "xgboost"):
        raise ValueError(f"Unsupported proxy engine: {engine}")
    started = time.monotonic()
    settings = json.loads(
        (STAGE22 / "configs" / "production.json").read_text(encoding="utf-8")
    )
    selected = columns if columns is not None else list(pre.columns)
    valid_semi_time = (semi.index < cutoff) & (
        semi.index >= cutoff - pd.Timedelta(days=lookback_days)
    )
    p_x = pre[selected].to_numpy(dtype="float32")
    s_x = semi.loc[valid_semi_time, selected].to_numpy(dtype="float32")
    x = np.vstack([p_x, s_x])
    y = np.vstack([pre_y, semi_y[valid_semi_time]])
    weights = np.concatenate(
        [np.full(len(pre), 0.25), np.ones(int(valid_semi_time.sum()))]
    )
    times = pre.index.append(semi.index[valid_semi_time])
    valid = (
        np.isfinite(y).all(axis=1)
        & (y > 0).all(axis=1)
        & (np.isfinite(x).mean(axis=1) >= 0.5)
    )
    x, y, weights, times = x[valid], y[valid], weights[valid], times[valid]
    if half_life_days:
        age_days = (cutoff - times).total_seconds().to_numpy() / 86400
        weights *= np.exp2(-age_days / half_life_days)
    active = np.isfinite(x).any(axis=0)
    result_dir.mkdir(parents=True, exist_ok=True)
    models, centers = [], []
    xgb_params = {
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": settings["tree"]["num_leaves"],
        "min_child_weight": settings["tree"]["min_data_in_leaf"],
        "max_bin": settings["tree"]["max_bin"],
        "reg_lambda": settings["tree"]["lambda_l2"],
        "colsample_bytree": settings["tree"]["feature_fraction"],
        "eta": settings["tree"]["learning_rate"],
        "nthread": settings["threads"],
        "seed": settings["seed"],
        "base_score": 0.0,
        "validate_parameters": True,
    }
    for target in range(2):
        center = float(np.median(y[:, target]))
        training_weights = weights / y[:, target]
        matrix = np.ascontiguousarray(x[:, active])
        if engine == "xgboost":
            dataset = xgb.DMatrix(
                matrix,
                label=y[:, target] - center,
                weight=training_weights / training_weights.mean(),
                nthread=settings["threads"],
            )
            model = xgb.train(
                xgb_params, dataset, num_boost_round=settings["proxy"]["rounds"]
            )
            model.save_model(str(result_dir / f"target_{target}.json"))
        else:
            params = {
                **settings["tree"],
                "objective": "regression_l1",
                "metric": "None",
                "verbosity": -1,
                "num_threads": settings["threads"],
                "seed": settings["seed"],
                "deterministic": True,
                "force_col_wise": True,
            }
            dataset = lgb.Dataset(
                matrix,
                label=y[:, target] - center,
                weight=training_weights / training_weights.mean(),
                free_raw_data=True,
            )
            model = lgb.train(
                params, dataset, num_boost_round=settings["proxy"]["rounds"]
            )
            model.save_model(str(result_dir / f"target_{target}.txt"))
        models.append(model)
        centers.append(center)
    (result_dir / "fit.json").write_text(
        json.dumps(
            {
                "cutoff": str(cutoff),
                "rows": len(x),
                "pre_rows": int(valid[: len(pre)].sum()),
                "features": selected,
                "active": np.flatnonzero(active).tolist(),
                "centers": centers,
                "half_life_days": half_life_days,
                "lookback_days": lookback_days,
                "engine": engine,
                "engine_version": xgb.__version__
                if engine == "xgboost"
                else lgb.__version__,
                "params": xgb_params if engine == "xgboost" else params,
                "rounds": settings["proxy"]["rounds"],
                "training_seconds": time.monotonic() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ProxyFit(
        models,
        centers,
        selected,
        active,
        pd.DataFrame(),
        cutoff,
        time.monotonic() - started,
        engine,
    )


def predict_curve(
    fit: ProxyFit,
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    contract: dict,
    *,
    day_weight: float = 0.0,
) -> np.ndarray:
    fit.history = history_only(truth, fit.cutoff)
    return predict_frozen(fit, frame, contract, day_weight=day_weight)


def predict_frozen(
    fit: ProxyFit,
    frame: pd.DataFrame,
    contract: dict,
    *,
    day_weight: float = 0.0,
) -> np.ndarray:
    """Inference receives only process features and an already-frozen history."""
    week = template(fit.history, frame.index, contract, "week4")
    if day_weight:
        day = template(fit.history, frame.index, contract, "day7")
        week = (1 - day_weight) * week + day_weight * day
    current = fit.current(frame)
    hours = (np.arange(96) * 15 + 7.5) / 60
    return reconcile(
        week + (current - week[:, 0, :])[:, None, :] * np.exp(-hours / 6)[None, :, None]
    )


def evaluate(
    name: str,
    fit: ProxyFit,
    validation: pd.DataFrame,
    bundle,
    result_dir: Path,
    *,
    day_weight: float = 0.0,
) -> dict:
    started = time.monotonic()
    index = validation.loc[CUTOFF:END].index
    pred = predict_curve(
        fit, validation.loc[index], bundle.truth, bundle.contract, day_weight=day_weight
    )
    actual = future_truth(bundle.truth, index, bundle.contract)
    metrics, summary = score(pred, actual)
    metrics.insert(0, "variant", name)
    metrics.to_csv(result_dir / f"{name}_metrics.csv", index=False)
    horizons = score_by_horizon(pred, actual)
    horizons.insert(0, "variant", name)
    horizons.to_csv(result_dir / f"{name}_by_horizon.csv", index=False)
    summary.update(
        {
            "variant": name,
            "origins": len(index),
            "prediction_seconds": time.monotonic() - started,
            "training_seconds": fit.training_seconds,
        }
    )
    (result_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run(data_dir: Path, result_dir: Path, *, engine: str = "xgboost") -> list[dict]:
    bundle = load_bundle(data_dir)
    pre, semi, pre_y, semi_y = load_feature_tables(data_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []

    fit_b1 = fit_proxy(
        pre, semi, pre_y, semi_y, CUTOFF, result_dir / "b1_model", engine=engine
    )
    results.append(evaluate("B1_physical_time", fit_b1, semi, bundle, result_dir))

    # B3 is one factor relative to B1: introduce a daily-period template.
    results.append(
        evaluate("B3_day_period", fit_b1, semi, bundle, result_dir, day_weight=0.5)
    )

    # One factor versus B3: include every May-Aug semifinal observation.
    fit_b7 = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        CUTOFF,
        result_dir / "b7_model",
        lookback_days=180,
        engine=engine,
    )
    results.append(
        evaluate(
            "B7_full_semifinal_history",
            fit_b7,
            semi,
            bundle,
            result_dir,
            day_weight=0.5,
        )
    )

    if engine == "xgboost":
        pd.DataFrame(results).to_csv(result_dir / "comparison.csv", index=False)
        return results

    pre_mechanism, semi_mechanism = add_mechanism(pre), add_mechanism(semi)
    fit_b4 = fit_proxy(
        pre_mechanism,
        semi_mechanism,
        pre_y,
        semi_y,
        CUTOFF,
        result_dir / "b4_model",
        engine=engine,
    )
    results.append(evaluate("B4_mechanism", fit_b4, semi_mechanism, bundle, result_dir))

    fit_b5 = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        CUTOFF,
        result_dir / "b5_model",
        half_life_days=120,
        engine=engine,
    )
    results.append(evaluate("B5_recency_120d", fit_b5, semi, bundle, result_dir))

    gain = np.zeros(len(fit_b1.columns), dtype="float64")
    for model in fit_b1.models:
        gain[fit_b1.active] += model.feature_importance(importance_type="gain")
    top = np.argsort(-gain)[:80]
    selected = [fit_b1.columns[i] for i in sorted(top)]
    fit_b6 = fit_proxy(
        pre,
        semi,
        pre_y,
        semi_y,
        CUTOFF,
        result_dir / "b6_model",
        columns=selected,
        engine=engine,
    )
    results.append(evaluate("B6_top80_gain", fit_b6, semi, bundle, result_dir))
    pd.DataFrame({"feature": fit_b1.columns, "gain": gain}).sort_values(
        "gain", ascending=False
    ).to_csv(result_dir / "b1_feature_gain.csv", index=False)
    pd.DataFrame(results).to_csv(result_dir / "comparison.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "features_experiments" / "exp_bc" / "data"
    )
    parser.add_argument("--engine", choices=("lightgbm", "xgboost"), default="xgboost")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    output = args.results or (
        ROOT
        / "results"
        / ("exp_bc_xgboost" if args.engine == "xgboost" else "exp_bc")
        / "b_variants"
    )
    print(
        json.dumps(
            run(args.data, output, engine=args.engine), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
