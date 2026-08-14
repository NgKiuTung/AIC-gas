"""Create an immutable local-only archive from a Git tag plus explicit private artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "results" / "releases"
PHASE1_PRIVATE_PATHS = (
    ROOT
    / "results"
    / "experiments"
    / "nested_temporal_validation"
    / "nested_outer_predictions_row_level.csv",
    ROOT / "results" / "evaluation" / "logs" / "nested_temporal_validation.log",
)
PHASE2_PRIVATE_PATHS = (
    ROOT / "results" / "models" / "production_forecaster",
    ROOT / "results" / "training" / "logs" / "production_refit.log",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tag_commit(version: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{version}^{{}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    )
    return completed.stdout.strip()


def tag_contains(version: str, path: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{version}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def expand_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        else:
            raise FileNotFoundError(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--set-current", action="store_true")
    args = parser.parse_args()
    commit = tag_commit(args.version)
    destination = RELEASES / args.version
    if destination.exists():
        raise FileExistsError(f"Private archives are immutable: {destination}")
    artifacts = destination / "artifacts"
    manifests = destination / "manifests"
    artifacts.mkdir(parents=True)
    manifests.mkdir(parents=True)

    repository_archive = artifacts / "repository_snapshot.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={repository_archive}", args.version],
        cwd=ROOT,
        check=True,
    )
    private_paths = list(PHASE1_PRIVATE_PATHS)
    includes_production = tag_contains(
        args.version, "results/production_validation/production_model_manifest.json"
    )
    if includes_production:
        private_paths.extend(PHASE2_PRIVATE_PATHS)
    private_files = expand_files(tuple(private_paths))
    private_archive = artifacts / "private_artifacts.zip"
    inventory: list[dict[str, object]] = []
    with zipfile.ZipFile(private_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in private_files:
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            archive.write(path, arcname=relative)
            inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (manifests / "private_artifact_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "version": args.version,
        "git_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "storage": "local_only_gitignored",
        "external_scoring_data_accessed": False,
        "dataset_included": False,
        "repository_snapshot": "artifacts/repository_snapshot.zip",
        "private_artifacts": "artifacts/private_artifacts.zip",
        "private_artifact_files": len(inventory),
        "production_models_included": includes_production,
    }
    (destination / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    print(
        f"Created private tag archive {args.version}: commit={commit}, "
        f"private_files={len(inventory)}, production_models={includes_production}"
    )


if __name__ == "__main__":
    main()
