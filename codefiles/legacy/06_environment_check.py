"""Record the modeling environment without accessing any dataset."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "environment"


def module_info(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        info = {"installed": True, "version": getattr(module, "__version__", "unknown")}
        if name == "torch":
            info.update(
                {
                    "cuda_available": bool(module.cuda.is_available()),
                    "cuda_version": module.version.cuda,
                    "gpu_count": int(module.cuda.device_count()),
                    "gpu_names": [module.cuda.get_device_name(i) for i in range(module.cuda.device_count())],
                }
            )
        return info
    except Exception as exc:  # environment audit must capture rather than hide failures
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "modules": {
            name: module_info(name)
            for name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "torch")
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    output = OUTPUT_DIR / "modeling_environment.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
