"""Hash and register every Phase 4 runtime and packaging artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "phase4_runtime_validation"
MANIFEST = RESULT_DIR / "phase4_artifact_manifest.json"
ARTIFACTS = (
    RESULT_DIR / "python310_container_validation_summary.json",
    RESULT_DIR / "python310_container_validation.log",
    RESULT_DIR / "synthetic_package_validation_summary.json",
    RESULT_DIR / "synthetic_package_validation.log",
    RESULT_DIR / "synthetic_contract_test_gas_predict_prelim.zip",
    ROOT / "results" / "figures_safe" / "phase4" / "python310_runtime_validation.png",
    ROOT / "results" / "figures_safe" / "phase4" / "python310_runtime_validation.pdf",
    ROOT / "results" / "compliance" / "phase4_runtime_packaging_gate.json",
    ROOT / "docs" / "experimental_docs" / "forecasting" / "08_Python310隔离运行与提交打包验证实验报告.md",
    ROOT / "requirements" / "runtime-py310.txt",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [str(path) for path in ARTIFACTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 4 artifacts: {missing}")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "phase4_python310_runtime_and_packaging",
        "official_scoring_data_accessed": False,
        "formal_submission_created": False,
        "artifacts": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in ARTIFACTS
        ],
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Phase 4 artifact manifest written: {len(ARTIFACTS)} files")


if __name__ == "__main__":
    main()
