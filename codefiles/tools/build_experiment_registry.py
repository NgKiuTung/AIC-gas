"""Build the compact experiment registry from the numbered legacy scripts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "registry" / "experiment_registry.csv"
MANIFEST = ROOT / "results" / "registry" / "registry_manifest.json"


def stage_for(number: int) -> tuple[str, str, str]:
    if number <= 7:
        return "preprocessing", "results/preprocessing", "docs/experimental_docs/preprocessing"
    if number <= 23:
        return "forecasting", "results/experiments", "docs/experimental_docs/forecasting"
    if number <= 37:
        return "forecasting_optimization", "results/experiments", "docs/experimental_docs/forecasting"
    return "dispatch_optimization", "results/optimization", "docs/experimental_docs/optimization"


def main() -> None:
    rows: list[dict[str, object]] = []
    for script in sorted((ROOT / "codefiles" / "legacy").glob("[0-9][0-9]_*.py")):
        number = int(script.name[:2])
        stage, result_group, report_group = stage_for(number)
        rows.append(
            {
                "step": number,
                "stage": stage,
                "script": str(script.relative_to(ROOT)).replace("\\", "/"),
                "result_group": result_group,
                "report_group": report_group,
                "status": "completed",
                "data_scope": "training-only",
                "introduced_version": "v0.1.0-preliminary",
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": len(rows),
                "external_scoring_data_accessed": False,
                "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Registered {len(rows)} legacy experiment steps in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

