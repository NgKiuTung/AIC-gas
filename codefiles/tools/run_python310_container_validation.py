"""Run the v0.5 inference snapshot in an isolated Python 3.10 container."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
VERSION = "v0.5.0-inference"
RELEASE_DIR = ROOT / "results" / "releases" / VERSION
RESULT_DIR = ROOT / "results" / "phase4_runtime_validation"
FIGURE_DIR = ROOT / "results" / "figures_safe" / "phase4"
LOG_PATH = RESULT_DIR / "python310_container_validation.log"
IMAGE = "python:3.10-slim"
RUNTIME_SUPPLEMENTS = (
    ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv",
    ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_logging() -> logging.Logger:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("python310_container_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def safe_extract(archive_path: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            extracted.append(member.filename)
        archive.extractall(destination)
    return extracted


def docker_image_id() -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout.strip()


def plot_runtime(summary: dict[str, object]) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    phase3 = summary["inner_phase3_summary"]
    labels = ["Training feature replay", "Synthetic raw-table inference", "Container total incl. install/tests"]
    values = [
        float(phase3["training_replay"]["seconds"]),
        float(phase3["synthetic_end_to_end"]["seconds"]),
        float(summary["container_total_seconds"]),
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    bars = axis.barh(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.bar_label(bars, labels=[f"{value:.2f} s" for value in values], padding=4)
    axis.set_xlabel("Wall-clock seconds (log scale)")
    axis.set_xscale("log")
    axis.set_title("Python 3.10 isolated runtime validation")
    axis.grid(axis="x", alpha=0.25)
    paths = [FIGURE_DIR / "python310_runtime_validation.png", FIGURE_DIR / "python310_runtime_validation.pdf"]
    for path in paths:
        figure.savefig(path, dpi=180 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    logger = setup_logging()
    repository_archive = RELEASE_DIR / "artifacts" / "repository_snapshot.zip"
    private_archive = RELEASE_DIR / "artifacts" / "private_artifacts.zip"
    if not repository_archive.is_file() or not private_archive.is_file():
        raise FileNotFoundError(f"Missing immutable archive for {VERSION}")
    cache_root = ROOT / "results" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase4_py310_", dir=cache_root) as temporary:
        sandbox = Path(temporary)
        repository_members = safe_extract(repository_archive, sandbox)
        private_members = safe_extract(private_archive, sandbox)
        forbidden_members = [
            member
            for member in [*repository_members, *private_members]
            if member.startswith("dataset/") and member != "dataset/.gitkeep"
        ]
        if forbidden_members:
            raise ValueError(f"Competition data unexpectedly present in sandbox: {forbidden_members[:5]}")
        runtime_supplements: list[dict[str, object]] = []
        for source in RUNTIME_SUPPLEMENTS:
            if not source.is_file():
                raise FileNotFoundError(f"Missing training-derived runtime supplement: {source}")
            relative = source.relative_to(ROOT)
            destination = sandbox / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            runtime_supplements.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                    "scope": "training-derived; no raw or scoring rows",
                }
            )
        stale_phase3_results = sandbox / "results" / "phase3_inference_validation"
        if stale_phase3_results.is_dir():
            resolved_stale = stale_phase3_results.resolve()
            resolved_sandbox = sandbox.resolve()
            if resolved_sandbox not in resolved_stale.parents:
                raise ValueError("Unsafe stale Phase 3 result path")
            shutil.rmtree(resolved_stale)
        logger.info(
            "Sandbox prepared from %s: repository_members=%d private_members=%d forbidden_data_members=0",
            VERSION,
            len(repository_members),
            len(private_members),
        )
        inner_command = (
            "python -m pip install --disable-pip-version-check -r requirements/base.txt -r requirements/dev.txt "
            "'xgboost>=2.0,<4' && "
            "python -m pip install --disable-pip-version-check -e . && "
            "python -m pytest && "
            "python codefiles/pipelines/run_phase3_inference_validation.py"
        )
        pip_cache = cache_root / "docker_pip_cache"
        pip_cache.mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "run", "--rm", "--network=bridge",
            "-v", f"{sandbox}:/workspace", "-w", "/workspace",
            "-v", f"{pip_cache}:/root/.cache/pip",
            IMAGE, "bash", "-lc", inner_command,
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - started
        logger.info("Docker stdout follows:\n%s", completed.stdout)
        if completed.stderr:
            logger.info("Docker stderr follows:\n%s", completed.stderr)
        inner_summary_path = (
            sandbox
            / "results"
            / "phase3_inference_validation"
            / "phase3_inference_validation_summary.json"
        )
        inner_summary = (
            json.loads(inner_summary_path.read_text(encoding="utf-8")) if inner_summary_path.is_file() else None
        )
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "version": VERSION,
            "git_commit": "5386882516fb5a45af886a22841e79f8440fe7e6",
            "python_image": IMAGE,
            "python_image_id": docker_image_id(),
            "container_total_seconds": elapsed,
            "container_returncode": completed.returncode,
            "repository_snapshot_sha256": sha256(repository_archive),
            "private_artifacts_sha256": sha256(private_archive),
            "sandbox_repository_members": len(repository_members),
            "sandbox_private_members": len(private_members),
            "runtime_supplements": runtime_supplements,
            "forbidden_data_members": forbidden_members,
            "scoring_directory_mounted": False,
            "official_scoring_data_accessed": False,
            "inner_phase3_summary": inner_summary,
            "python310_full_validation_pass": bool(
                completed.returncode == 0 and inner_summary is not None and inner_summary["all_passed"]
            ),
        }
    if summary["inner_phase3_summary"] is not None:
        figure_paths = plot_runtime(summary)
    else:
        figure_paths = []
    summary["figures"] = [str(path.relative_to(ROOT)) for path in figure_paths]
    summary_path = RESULT_DIR / "python310_container_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Python 3.10 validation pass=%s total_seconds=%.3f", summary["python310_full_validation_pass"], elapsed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["python310_full_validation_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
