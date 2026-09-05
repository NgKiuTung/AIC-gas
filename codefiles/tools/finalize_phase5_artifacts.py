"""Hash and register Phase 5 final inference artifacts without publishing predictions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "phase5_final_adapter_validation"
MANIFEST = RESULT_DIR / "phase5_artifact_manifest.json"
ARTIFACTS = (
    (RESULT_DIR / "synthetic_final_adapter_rehearsal_summary.json", "safe"),
    (RESULT_DIR / "synthetic_final_adapter_rehearsal.log", "safe_log"),
    (RESULT_DIR / "synthetic_192row_rehearsal_gas_predict_prelim.zip", "local_synthetic_zip"),
    (ROOT / "results" / "figures_safe" / "phase5" / "synthetic_192row_final_adapter.png", "safe"),
    (ROOT / "results" / "figures_safe" / "phase5" / "synthetic_192row_final_adapter.pdf", "safe"),
    (ROOT / "results" / "compliance" / "final_scoring_authorization.json", "safe"),
    (
        ROOT / "results" / "final_inference_audit" / "final_scoring_inference_manifest.json",
        "safe_metadata",
    ),
    (
        ROOT / "results" / "final_inference_audit" / "final_scoring_inference.log",
        "safe_log",
    ),
    (
        ROOT / "results" / "final_inference_audit" / "final_submission_independent_audit.json",
        "safe_metadata",
    ),
    (
        ROOT / "results" / "final_inference_audit" / "submission_contract_hardening.json",
        "safe_metadata",
    ),
    (
        ROOT / "results" / "final_inference_audit" / "dual_csv_contract_rebuild.json",
        "safe_metadata",
    ),
    (
        ROOT / "results" / "final_inference_audit" / "final_submission_independent_audit.log",
        "safe_log",
    ),
    (
        ROOT / "results" / "submissions" / "final" / "AIC-gas_gas_predict_prelim.zip",
        "private_submission",
    ),
    (
        ROOT
        / "results"
        / "submissions"
        / "final"
        / "diagnostics"
        / "final_forecast_diagnostics.png",
        "private_prediction_figure",
    ),
    (
        ROOT
        / "docs"
        / "experimental_docs"
        / "forecasting"
        / "09_最终评分只读推理与提交验包实验报告.md",
        "safe",
    ),
    (
        ROOT
        / "docs"
        / "experimental_docs"
        / "forecasting"
        / "10_初赛正式提交契约加固与无损重打包报告.md",
        "safe",
    ),
    (
        ROOT
        / "docs"
        / "experimental_docs"
        / "forecasting"
        / "11_初赛双CSV正式提交包整改与验收报告.md",
        "safe",
    ),
    (
        ROOT
        / "results"
        / "repository_validation"
        / "release_v0.7.0-submission-rc1_verification.json",
        "safe_metadata",
    ),
    (
        ROOT
        / "results"
        / "repository_validation"
        / "release_v0.7.0-submission-rc1_verification_history.txt",
        "safe_log",
    ),
    (
        ROOT
        / "results"
        / "repository_validation"
        / "release_v0.7.0-submission-rc2_verification.json",
        "safe_metadata",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [str(path) for path, _ in ARTIFACTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 5 artifacts: {missing}")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase5_final_scoring_read_only_inference",
        "official_scoring_data_accessed": True,
        "training_or_tuning_performed": False,
        "label_metrics_computed": False,
        "private_artifacts_are_gitignored": True,
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "classification": classification,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path, classification in ARTIFACTS
        ],
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phase 5 artifact manifest written: {len(ARTIFACTS)} files")


if __name__ == "__main__":
    main()
