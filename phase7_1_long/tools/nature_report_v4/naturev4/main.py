"""CLI for the corrected Nature-Skills-driven AIC-Gas report figure set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fig_features import build as feature_figure
from .fig_pipeline import build as pipeline_figure
from .fig_quality import build as quality_figure
from .fig_score import build as score_figure
from .fig_selection import build as selection_figure
from .io import Paths

FIGURES = [pipeline_figure, feature_figure, selection_figure, score_figure, quality_figure]


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase7-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    phase7 = args.phase7_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths = Paths(repo=repo, phase7=phase7)
    qa = []
    for builder in FIGURES:
        result = builder(paths, out)
        if result is not None:
            qa.append(result)

    index = ["# Phase7.1 Nature-Skills Figure Index", ""]
    for item in qa:
        stem = item["figure"]
        index += [f"## {stem}", "", f"- PDF: `{stem}.pdf`", f"- SVG: `{stem}.svg`",
                  f"- TIFF: `{stem}.tiff`", f"- PNG: `{stem}.png`", f"- QA: `{stem}.qa.json`", ""]
    (out / "FIGURE_INDEX.md").write_text("\n".join(index), encoding="utf-8")
    (out / "QA_SUMMARY.json").write_text(json.dumps({"figures": qa}, indent=2), encoding="utf-8")
    print(f"generated={len(qa)}")
    print(f"out={out}")


if __name__ == "__main__":
    cli()
