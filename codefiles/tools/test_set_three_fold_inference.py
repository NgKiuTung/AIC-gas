"""Run chronological three-block inference on the unlabeled scoring tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "codefiles" / "pipelines"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import run_final_submission_inference as final_inference  # noqa: E402

from gas_power.data.causal_preprocessing import preprocess_causal_raw_tables  # noqa: E402
from gas_power.features.inference import build_inference_feature_frame, select_model_features  # noqa: E402
from gas_power.forecasting.production import apply_frozen_ensemble  # noqa: E402
import yaml  # noqa: E402


OUTPUT_DIR = ROOT / "results" / "experiments" / "test_set_three_fold_inference"
HORIZONS = tuple(range(1, 9))


def main() -> None:
    training = final_inference.load_csv_tables(final_inference.TRAINING_DIR, scoring=False)
    scoring = final_inference.load_csv_tables(final_inference.SCORING_DIR, scoring=True)
    combined, scoring_times = final_inference.build_combined_tables(training, scoring)
    causal, _ = preprocess_causal_raw_tables(
        combined,
        final_inference.load_price_lookup(),
        split="three_fold_test_inference",
    )
    features = build_inference_feature_frame(causal)
    schema = final_inference.load_frozen_schema()
    test_features = features.loc[features["datetime"].isin(scoring_times)].sort_values("datetime")
    matrix = select_model_features(test_features, schema).to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Three-fold model matrix contains NaN or Inf")

    component_raw = final_inference.load_component_predictions(matrix)
    spec = yaml.safe_load(final_inference.MODEL_SPEC_PATH.read_text(encoding="utf-8"))
    current_1 = test_features["feat_p50_current"].to_numpy(dtype=float)
    current_all = (
        test_features["feat_p50_current"].to_numpy(dtype=float)
        + test_features["feat_p120_current"].to_numpy(dtype=float)
    )
    prediction = apply_frozen_ensemble(
        component_raw,
        current_1,
        current_all,
        spec["ensemble_parameters"],
    )
    predicted = np.concatenate([prediction["generator_1"], prediction["generator_all"]], axis=1)
    fold_indices = np.array_split(np.arange(len(test_features)), 3)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for fold_number, indices in enumerate(fold_indices, start=1):
        fold_values = predicted[indices]
        hierarchy = bool(
            (prediction["generator_all"][indices] >= prediction["generator_1"][indices]).all()
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "rows": len(indices),
                "start": str(test_features.iloc[indices[0]]["datetime"]),
                "end": str(test_features.iloc[indices[-1]]["datetime"]),
                "prediction_min": float(fold_values.min()),
                "prediction_max": float(fold_values.max()),
                "prediction_mean": float(fold_values.mean()),
                "prediction_std": float(fold_values.std()),
                "finite_predictions": bool(np.isfinite(fold_values).all()),
                "generator_hierarchy": hierarchy,
                "mape": None,
                "mape_status": "unavailable_without_target_labels",
            }
        )
        row_frame = pd.DataFrame({"datetime": test_features.iloc[indices]["datetime"].to_numpy()})
        for target in ("generator_1", "generator_all"):
            for horizon_index, horizon in enumerate(HORIZONS):
                row_frame[f"{target}_t_plus_{horizon * 15}m"] = prediction[target][indices, horizon_index]
        row_frame.insert(1, "fold", fold_number)
        prediction_rows.append(row_frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "three_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "data_source": str(final_inference.SCORING_DIR),
        "evaluation_type": "chronological_three_block_inference_without_labels",
        "official_scoring_data_accessed": True,
        "fold_count": 3,
        "rows": len(test_features),
        "model_feature_count": len(schema),
        "mape": None,
        "mape_status": "unavailable_without_target_labels",
        "all_predictions_finite": bool(np.isfinite(predicted).all()),
        "all_folds_hierarchy_valid": bool(fold_summary["generator_hierarchy"].all()),
        "folds": fold_rows,
        "output_predictions": str((OUTPUT_DIR / "three_fold_predictions.csv").relative_to(ROOT)),
        "output_summary": str((OUTPUT_DIR / "fold_summary.csv").relative_to(ROOT)),
    }
    (OUTPUT_DIR / "three_fold_inference_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
