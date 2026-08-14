"""Stable entrypoint for the recorded forecasting experiment sequence."""

import argparse

import _bootstrap  # noqa: F401
from gas_power.utils.pipeline import run_legacy_scripts

SCRIPT_NAMES = {
    8: "baseline_experiments", 9: "xgboost_gpu_experiments", 10: "xgboost_crossfold_calibration",
    11: "xgboost_direct_targets", 12: "recency_window_screen", 13: "oof_ensemble_and_shrinkage",
    14: "forecast_robustness_analysis", 15: "build_time_augmented_features", 16: "time_feature_ablation",
    17: "temporal_effect_analysis", 18: "relative_residual_ablation", 19: "hybrid_target_selection",
    20: "train_final_forecaster", 22: "final_hybrid_error_analysis", 23: "validate_modeling_stage",
    24: "cleaning_reaudit", 25: "build_cleaning_enhanced_features", 26: "cleaning_enhanced_hybrid_cv",
    27: "cleaning_model_oof_ensemble", 28: "mape_aligned_objectives", 29: "tcn_sequence_model",
    30: "cleaning_feature_importance", 31: "cleaning_feature_group_ablation", 32: "dynamic_shrinkage_gate",
    33: "generator1_mode_classifier", 34: "targeted_xgboost_capacity", 35: "multidepth_dynamic_ensemble",
    36: "build_known_future_price_features", 37: "known_future_price_cv",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", type=int, default=8)
    parser.add_argument("--to-step", type=int, default=37)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [f"{n:02d}_{SCRIPT_NAMES[n]}.py" for n in sorted(SCRIPT_NAMES) if args.from_step <= n <= args.to_step]
    run_legacy_scripts(selected, "training", args.dry_run)

