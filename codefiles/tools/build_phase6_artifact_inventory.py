"""Record hashes of local-only Phase 6 artifacts without copying their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

DATA_FILES = (
    "features_experiments/exp_bc/data/preliminary_clean_15m.parquet",
    "features_experiments/exp_bc/data/semifinal_clean_1m.parquet",
    "features_experiments/exp_bc/data/train_common_jan_sep_15m.parquet",
    "features_experiments/exp_bc/data/common_features_preliminary_15m.parquet",
    "features_experiments/exp_bc/data/common_features_semifinal_15m.parquet",
    "features_experiments/exp_bc/data/semifinal_highfreq_at_15m_origins.parquet",
    "features_experiments/exp_bc/data/feature_manifest.json",
    "wjt/gas_stage22_rebuild/features/history_process_features.npz",
)
RESULTS_DIR = Path("results/exp_bc_xgboost")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(source_root: Path) -> dict:
    source_root = source_root.resolve(strict=True)
    result_root = (source_root / RESULTS_DIR).resolve(strict=True)
    if not result_root.is_relative_to(source_root):
        raise ValueError("Results directory resolves outside source root")
    paths = [Path(relative) for relative in DATA_FILES]
    paths.extend(path.relative_to(source_root) for path in sorted(result_root.rglob("*")) if path.is_file())
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact path")

    artifacts = []
    for relative in paths:
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"Invalid local artifact: {relative}")
        artifacts.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
                "publication": "local_only",
            }
        )
    return {
        "schema_version": 1,
        "scope": "phase6_xgboost_process_proxy",
        "contains_artifact_values": False,
        "raw_competition_sources_enumerated": False,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_inventory(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Recorded {len(manifest['artifacts'])} local-only artifact hashes")


if __name__ == "__main__":
    main()
