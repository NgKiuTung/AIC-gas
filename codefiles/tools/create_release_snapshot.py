"""Create an immutable, checksum-verified local milestone snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "results" / "releases"
VERSION_PATTERN = re.compile(r"v\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?$")
FORBIDDEN_DIRECTORY_NAME = "初赛-评分所用测试集"
ARCHIVE_EXCLUDED_RESULT_DIRECTORIES = {"releases", "submissions"}

TRACKED_FILES = {
    "metrics/forecast_summary.json": "results/experiments/multidepth_ensemble/experiment_summary.json",
    "metrics/dispatch_summary.json": "results/optimization/dispatch_validation/selected_scenario_and_acceptance.json",
    "manifests/optimization_validation.json": "results/optimization/final_validation/validation_summary.json",
    "manifests/optimization_pipeline.json": "results/optimization/pipeline_run_manifest.json",
    "artifacts/data/forecast_selected_parameters.csv": (
        "results/experiments/multidepth_ensemble/selected_parameters.csv"
    ),
    "artifacts/data/forecast_fold_summary.csv": "results/experiments/multidepth_ensemble/fold_summary.csv",
    "artifacts/data/forecast_selected_oof_predictions.csv": (
        "results/experiments/multidepth_ensemble/selected_oof_predictions.csv"
    ),
    "artifacts/data/dispatch_selected_schedules.csv": (
        "results/optimization/dispatch_validation/selected_balanced_schedules.csv"
    ),
    "artifacts/data/dispatch_fold_summary.csv": (
        "results/optimization/dispatch_validation/selected_scenario_by_fold.csv"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def full_archive_sources() -> list[Path]:
    """Return experiment artifacts and reports without traversing dataset or releases."""
    sources: list[Path] = []
    results_root = ROOT / "results"
    for child in sorted(results_root.iterdir()):
        if child.name in ARCHIVE_EXCLUDED_RESULT_DIRECTORIES:
            continue
        if child.is_file():
            sources.append(child)
        elif child.is_dir():
            sources.extend(sorted(path for path in child.rglob("*") if path.is_file()))
    reports_root = ROOT / "docs" / "experimental_docs"
    sources.extend(sorted(path for path in reports_root.rglob("*") if path.is_file()))
    if any(FORBIDDEN_DIRECTORY_NAME in str(path) or "dataset" in path.relative_to(ROOT).parts for path in sources):
        raise RuntimeError("Forbidden dataset path entered the archive source list")
    return sources


def create_full_experiment_archive(destination: Path) -> dict[str, object]:
    """Create one compressed, locally retained rollback package and its tracked inventory."""
    archive_path = destination / "artifacts" / "full_experiment_archive.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    sources = full_archive_sources()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for index, source in enumerate(sources, start=1):
            relative = str(source.relative_to(ROOT)).replace("\\", "/")
            digest = hashlib.sha256()
            info = zipfile.ZipInfo.from_file(source, arcname=relative)
            info.compress_type = zipfile.ZIP_DEFLATED
            with source.open("rb") as input_handle, archive.open(info, "w") as output_handle:
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output_handle.write(chunk)
            inventory.append({"path": relative, "bytes": source.stat().st_size, "sha256": digest.hexdigest()})
            if index % 50 == 0 or index == len(sources):
                print(f"Archived experiment files: {index}/{len(sources)}", flush=True)
    inventory_path = destination / "manifests" / "full_archive_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "archive": str(archive_path.relative_to(destination)).replace("\\", "/"),
        "inventory": str(inventory_path.relative_to(destination)).replace("\\", "/"),
        "source_files": len(inventory),
        "source_bytes": sum(int(row["bytes"]) for row in inventory),
        "archive_bytes": archive_path.stat().st_size,
        "excluded_result_directories": sorted(ARCHIVE_EXCLUDED_RESULT_DIRECTORIES),
        "dataset_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--set-current", action="store_true")
    args = parser.parse_args()
    if not VERSION_PATTERN.fullmatch(args.version):
        raise ValueError("Version must match vMAJOR.MINOR.PATCH[-label]")

    destination = RELEASES / args.version
    if destination.exists():
        raise FileExistsError(f"Release snapshots are immutable: {destination}")
    destination.mkdir(parents=True)

    copied: list[dict[str, str]] = []
    for target_name, source_name in TRACKED_FILES.items():
        source = ROOT / source_name
        target = destination / target_name
        copy_file(source, target)
        copied.append({"source": source_name, "snapshot": target_name})

    for config in sorted((ROOT / "configs").glob("*.yaml")):
        target = destination / "configs" / config.name
        copy_file(config, target)
        copied.append(
            {
                "source": str(config.relative_to(ROOT)).replace("\\", "/"),
                "snapshot": f"configs/{config.name}",
            }
        )

    reports_source = ROOT / "docs" / "experimental_docs"
    reports_target = destination / "reports" / "experimental_docs"
    if not reports_source.is_dir():
        raise FileNotFoundError(reports_source)
    shutil.copytree(reports_source, reports_target)
    copied.append(
        {"source": "docs/experimental_docs", "snapshot": "reports/experimental_docs"}
    )

    for group in ("prediction", "optimization"):
        source_dir = ROOT / "results" / "visualizations" / group
        for figure in sorted(source_dir.glob("*.*")):
            if figure.suffix.lower() not in {".png", ".pdf", ".svg", ".json"}:
                continue
            target = destination / "figures" / group / figure.name
            copy_file(figure, target)
            copied.append(
                {
                    "source": str(figure.relative_to(ROOT)).replace("\\", "/"),
                    "snapshot": str(target.relative_to(destination)).replace("\\", "/"),
                }
            )

    model_source = ROOT / "results" / "models" / "power_forecaster"
    model_target = destination / "artifacts" / "models" / "power_forecaster"
    if not model_source.is_dir():
        raise FileNotFoundError(model_source)
    shutil.copytree(model_source, model_target)
    copied.append({"source": "results/models/power_forecaster", "snapshot": "artifacts/models/power_forecaster"})

    full_archive = create_full_experiment_archive(destination)

    checksum_rows = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        checksum_rows.append(
            {
                "path": str(path.relative_to(destination)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (destination / "checksums.json").write_text(
        json.dumps(checksum_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "source_scope": "training-only derived artifacts",
        "external_scoring_data_accessed": False,
        "raw_training_data_copied": False,
        "experiment_reports_copied": True,
        "full_experiment_archive": full_archive,
        "files_before_checksum_manifest": len(checksum_rows),
        "bytes_before_checksum_manifest": sum(row["bytes"] for row in checksum_rows),
        "copies": copied,
        "rollback": {
            "code": f"git checkout {args.version} after the matching tag is created",
            "artifacts": f"results/releases/{args.version}/artifacts",
            "integrity": f"results/releases/{args.version}/checksums.json",
        },
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.set_current:
        current = RELEASES / "current"
        current.mkdir(parents=True, exist_ok=True)
        (current / "version.json").write_text(
            json.dumps(
                {
                    "version": args.version,
                    "release_manifest": f"results/releases/{args.version}/release_manifest.json",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Created immutable release {args.version}: {len(checksum_rows)} files")


if __name__ == "__main__":
    main()
