"""Verify every archived file listed in a release checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    release = ROOT / "results" / "releases" / args.version
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    scoring_accessed = manifest.get("external_scoring_data_accessed")
    if not isinstance(scoring_accessed, bool):
        raise RuntimeError("Release data-scope declaration is missing or invalid")
    if manifest.get("dataset_included") is not False:
        raise RuntimeError("Release must explicitly declare dataset_included=false")
    formal_submission = bool(manifest.get("formal_submission_included", False))
    if formal_submission and not scoring_accessed:
        raise RuntimeError("A formal submission archive must declare scoring access")
    rows = json.loads((release / "checksums.json").read_text(encoding="utf-8"))
    failures = []
    for row in rows:
        path = release / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    archive_summary = manifest.get("full_experiment_archive")
    archive_error = None
    archive_members = 0
    if archive_summary:
        archive_path = release / archive_summary["archive"]
        inventory_path = release / archive_summary["inventory"]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = len(archive.infolist())
            archive_error = archive.testzip()
        if archive_members != len(inventory):
            archive_error = f"member_count={archive_members}; inventory_count={len(inventory)}"
        if any(row["path"].startswith("dataset/") for row in inventory):
            archive_error = "dataset path found in full archive inventory"
    private_archive = release / manifest["private_artifacts"]
    private_inventory_path = release / "manifests" / "private_artifact_inventory.json"
    private_inventory = json.loads(private_inventory_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(private_archive, "r") as archive:
        private_members = [item.filename for item in archive.infolist()]
        private_archive_error = archive.testzip()
    expected_private_members = [row["path"] for row in private_inventory]
    if private_members != expected_private_members:
        private_archive_error = "private archive member list differs from inventory"
    if any(path.startswith("dataset/") for path in private_members):
        private_archive_error = "dataset path found in private archive"
    if formal_submission and not any(path.startswith("results/submissions/") for path in private_members):
        private_archive_error = "formal submission declared but missing from private archive"
    result = {
        "version": args.version,
        "checked_files": len(rows),
        "failed_files": failures,
        "full_archive_members": archive_members,
        "full_archive_error": archive_error,
        "private_archive_members": len(private_members),
        "private_archive_error": private_archive_error,
        "dataset_included": False,
        "formal_submission_included": formal_submission,
        "overall_pass": not failures and archive_error is None and private_archive_error is None,
        "external_scoring_data_accessed": scoring_accessed,
    }
    output = ROOT / "results" / "repository_validation" / f"release_{args.version}_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Release verification: {len(rows) - len(failures)}/{len(rows)} files passed")
    if failures or archive_error is not None or private_archive_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
