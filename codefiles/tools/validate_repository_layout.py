"""Validate repository structure without traversing any dataset directory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "repository_validation"


def add(checks: list[dict[str, object]], name: str, passed: bool, evidence: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    checks: list[dict[str, object]] = []
    required_directories = [
        "codefiles/src/gas_power",
        "codefiles/pipelines",
        "codefiles/legacy",
        "codefiles/tests",
        "codefiles/tools",
        "configs",
        "docs/design",
        "docs/experimental_docs/preprocessing",
        "docs/experimental_docs/forecasting",
        "docs/experimental_docs/optimization",
        "results/registry",
    ]
    add(
        checks,
        "required_directories",
        all((ROOT / path).is_dir() for path in required_directories),
        ", ".join(required_directories),
    )
    legacy_scripts = sorted((ROOT / "codefiles" / "legacy").glob("[0-9][0-9]_*.py"))
    add(checks, "all_45_legacy_scripts", len(legacy_scripts) == 45, f"count={len(legacy_scripts)}")
    add(
        checks,
        "legacy_roots_migrated",
        all("parents[1]" not in path.read_text(encoding="utf-8") for path in legacy_scripts),
        "all numbered scripts resolve project root from legacy depth",
    )
    root_python = list(ROOT.glob("*.py"))
    add(checks, "all_python_under_codefiles", not root_python, f"root_python_count={len(root_python)}")
    root_documents = [
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".pdf", ".docx"}
    ]
    add(checks, "root_document_exception_only", root_documents == ["README.md"], f"root_documents={root_documents}")
    add(
        checks,
        "dataset_ignored_by_policy",
        "dataset/*" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
        "dataset was not traversed",
    )
    registry = ROOT / "results" / "registry" / "experiment_registry.csv"
    add(checks, "experiment_registry_exists", registry.is_file(), str(registry.relative_to(ROOT)))
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    add(
        checks,
        "local_releases_excluded_from_git",
        "results/releases/" in gitignore,
        "version archives remain local and are not Git-tracked",
    )

    overall = all(bool(item["passed"]) for item in checks)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_scoring_data_accessed": False,
        "dataset_directory_traversed": False,
        "checks": checks,
        "passed": sum(bool(item["passed"]) for item in checks),
        "failed": sum(not bool(item["passed"]) for item in checks),
        "overall_pass": overall,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "structure_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [f"Repository validation: {summary['passed']}/{len(checks)} passed"]
    lines.extend(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['evidence']}" for item in checks)
    (OUTPUT_DIR / "structure_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[0])
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
