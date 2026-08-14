"""Run the preregistered training-only nested temporal pseudo-test protocol."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml
from gas_power.evaluation.nested_temporal import (
    build_candidate_family,
    mape,
    predict_candidate,
    select_candidate,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "evaluation_protocol.yaml"
D5_PATH = ROOT / "results" / "experiments" / "cleaning_group_ablation" / "raw_oof_predictions.csv"
CAPACITY_PATH = ROOT / "results" / "experiments" / "targeted_xgb_capacity" / "raw_oof_predictions.csv"
RESULT_DIR = ROOT / "results" / "experiments" / "nested_temporal_validation"
LOG_DIR = ROOT / "results" / "evaluation" / "logs"
MODEL_SPEC_PATH = ROOT / "configs" / "model_spec.yaml"
KEYS = ["datetime", "fold", "target", "horizon_step", "horizon_minutes"]


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nested_temporal_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "nested_temporal_validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_components() -> pd.DataFrame:
    d5_all = pd.read_csv(D5_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    d5 = d5_all[d5_all["variant"] == "core_smooth_transition_ratio_clean"].copy()
    capacity = pd.read_csv(CAPACITY_PATH, encoding="utf-8-sig", parse_dates=["datetime"])
    d4 = capacity[capacity["config"] == "d4_wide"].copy()
    d6 = capacity[capacity["config"] == "d6_regularized"].copy()
    for name, frame in (("d5", d5), ("d4", d4), ("d6", d6)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate component rows in {name}")
    reference = d5[KEYS + ["actual", "current", "raw_correction"]].rename(
        columns={"raw_correction": "correction_d5"}
    )
    for name, frame in (("d4", d4), ("d6", d6)):
        truth = frame[KEYS + ["actual", "current"]]
        checked = reference[KEYS + ["actual", "current"]].merge(
            truth,
            on=KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", f"_{name}"),
        )
        if len(checked) != len(reference):
            raise ValueError(f"Component key mismatch for {name}")
        for column in ("actual", "current"):
            if not np.allclose(checked[f"{column}_reference"], checked[f"{column}_{name}"], rtol=0, atol=1e-6):
                raise ValueError(f"Component {column} mismatch for {name}")
        reference = reference.merge(
            frame[KEYS + ["raw_correction"]].rename(columns={"raw_correction": f"correction_{name}"}),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    numeric = reference[["actual", "current", "correction_d5", "correction_d4", "correction_d6"]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite values in component predictions")
    return reference.sort_values(KEYS).reset_index(drop=True)


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["actual"].to_numpy(dtype=np.float64),
        frame["current"].to_numpy(dtype=np.float64),
        frame[["correction_d5", "correction_d4", "correction_d6"]].to_numpy(dtype=np.float64),
        frame["horizon_step"].to_numpy(dtype=np.float64),
    )


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["official_alignment"]["scoring_data_allowed"]:
        raise ValueError("Nested validation must never access scoring data")
    components = load_components()
    candidates = build_candidate_family()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([candidate.to_dict() for candidate in candidates]).to_csv(
        RESULT_DIR / "candidate_family.csv", index=False, encoding="utf-8-sig"
    )
    LOGGER.info("Loaded %d training-only component rows and %d frozen candidates", len(components), len(candidates))

    selection_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    gap = pd.Timedelta(minutes=int(protocol["outer_gap_minutes"]))
    for window in protocol["outer_windows"]:
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        inner = components[components["datetime"] < start - gap]
        outer = components[(components["datetime"] >= start) & (components["datetime"] <= end)]
        if inner.empty or outer.empty:
            raise ValueError(f"Empty inner/outer data for {window['name']}")
        if not (inner["datetime"].max() < start - gap):
            raise ValueError(f"Temporal gap violation for {window['name']}")
        LOGGER.info(
            "%s inner=%s..%s rows=%d outer=%s..%s rows=%d",
            window["name"],
            inner["datetime"].min(),
            inner["datetime"].max(),
            len(inner),
            outer["datetime"].min(),
            outer["datetime"].max(),
            len(outer),
        )
        for target in ("generator_1", "generator_all"):
            inner_target = inner[inner["target"] == target]
            outer_target = outer[outer["target"] == target].copy()
            selected, inner_mape = select_candidate(*arrays(inner_target), candidates)
            outer_actual, outer_current, outer_corrections, outer_horizon = arrays(outer_target)
            prediction = predict_candidate(outer_current, outer_corrections, outer_horizon, selected)
            outer_target["prediction"] = prediction
            outer_target["ape_model"] = np.abs(prediction - outer_actual) / np.maximum(np.abs(outer_actual), 1e-6)
            outer_target["ape_persistence"] = np.abs(outer_current - outer_actual) / np.maximum(
                np.abs(outer_actual), 1e-6
            )
            outer_target["outer_window"] = window["name"]
            outer_target["selected_candidate_id"] = selected.candidate_id
            prediction_parts.append(outer_target)
            row = {
                "outer_window": window["name"],
                "outer_start": window["start"],
                "outer_end": window["end"],
                "target": target,
                "inner_rows": len(inner_target),
                "outer_rows": len(outer_target),
                "inner_cutoff_exclusive": str(start - gap),
                "inner_mape": inner_mape,
                "outer_model_mape": mape(outer_actual, prediction),
                "outer_persistence_mape": mape(outer_actual, outer_current),
                **selected.to_dict(),
            }
            selection_rows.append(row)
            LOGGER.info(
                "%s/%s selected=%s inner=%.4f%% outer=%.4f%% persistence=%.4f%%",
                window["name"],
                target,
                selected.candidate_id,
                inner_mape * 100,
                row["outer_model_mape"] * 100,
                row["outer_persistence_mape"] * 100,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    selection = pd.DataFrame(selection_rows)
    metrics = predictions.groupby(
        ["outer_window", "target", "horizon_step", "horizon_minutes"], as_index=False
    ).agg(persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean"))
    window_summary = metrics.groupby("outer_window", as_index=False).agg(
        persistence_mape=("persistence_mape", "mean"), model_mape=("model_mape", "mean")
    )
    window_summary["score"] = 1.0 - window_summary["model_mape"]
    window_summary["absolute_improvement_pp"] = (
        window_summary["persistence_mape"] - window_summary["model_mape"]
    ) * 100
    window_summary["relative_improvement"] = (
        window_summary["persistence_mape"] - window_summary["model_mape"]
    ) / window_summary["persistence_mape"]

    final_rows: list[dict[str, object]] = []
    for target in ("generator_1", "generator_all"):
        target_frame = components[components["target"] == target]
        selected, training_oof_mape = select_candidate(*arrays(target_frame), candidates)
        final_rows.append({"target": target, "training_oof_mape": training_oof_mape, **selected.to_dict()})
    final_selection = pd.DataFrame(final_rows)
    overall_model_mape = float(metrics["model_mape"].mean())
    overall_persistence_mape = float(metrics["persistence_mape"].mean())

    selection.to_csv(RESULT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "outer_metrics_by_target_horizon.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(RESULT_DIR / "outer_window_summary.csv", index=False, encoding="utf-8-sig")
    final_selection.to_csv(RESULT_DIR / "final_selected_parameters.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        RESULT_DIR / "nested_outer_predictions_row_level.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "evaluation_type": "retrospective_nested_temporal_pseudotest",
        "historical_architecture_selection_caveat": True,
        "external_scoring_data_accessed": False,
        "component_rows": len(components),
        "candidate_count": len(candidates),
        "outer_windows": len(protocol["outer_windows"]),
        "nested_outer_mape": overall_model_mape,
        "nested_outer_score": 1.0 - overall_model_mape,
        "nested_outer_persistence_mape": overall_persistence_mape,
        "relative_improvement_vs_persistence": (
            overall_persistence_mape - overall_model_mape
        ) / overall_persistence_mape,
        "all_outer_windows_improve": bool((window_summary["model_mape"] < window_summary["persistence_mape"]).all()),
        "post_selection_oof_mape_reference": 0.054671132544103794,
        "final_selection": final_rows,
        "input_sha256": {"d5": sha256(D5_PATH), "capacity": sha256(CAPACITY_PATH)},
    }
    (RESULT_DIR / "nested_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_spec = {
        "model_spec_id": "gas_forecast_multidepth_v1",
        "status": "evaluation_frozen_pending_phase2_production_refit",
        "created_utc": report["created_utc"],
        "protocol_frozen_commit": protocol["frozen_commit"],
        "training_scope": "official preliminary training period only",
        "raw_observation_range": ["2025-01-01 00:00:00", "2025-04-30 23:45:00"],
        "supervised_origin_range": ["2025-01-01 00:00:00", "2025-04-30 21:45:00"],
        "external_scoring_data_accessed": False,
        "feature_version": "cleaning_enhanced_core_smooth_transition_ratio_clean",
        "target_design": {"generator_1": "relative_correction", "generator_all": "absolute_correction"},
        "component_models": {
            "d5": "350 trees, depth 5, cleaning-enhanced selected feature groups",
            "d4": "500 trees, depth 4, wide capacity candidate",
            "d6": "450 trees, depth 6, regularized capacity candidate",
        },
        "production_components_with_nonzero_weight": ["d5", "d6"],
        "ensemble_parameters": final_rows,
        "validation_protocol": protocol["protocol_id"],
        "reported_metrics": {
            "post_selection_oof_mape": report["post_selection_oof_mape_reference"],
            "retrospective_nested_outer_mape": overall_model_mape,
            "retrospective_nested_outer_score": 1.0 - overall_model_mape,
        },
        "production_bundle_available": False,
        "next_required_phase": "refit and serialize every non-zero component model, then verify reload parity",
    }
    MODEL_SPEC_PATH.write_text(yaml.safe_dump(model_spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    LOGGER.info("Summary: %s", json.dumps(report, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
