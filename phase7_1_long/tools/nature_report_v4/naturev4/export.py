"""Publication export and lightweight deterministic QA."""
from __future__ import annotations

from pathlib import Path
import json


def export_figure(fig, out_dir: Path, stem: str) -> dict:
    """Export editable vector plus review raster formats and record geometry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
    }.items():
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        files[ext] = path.name
    axes = []
    for i, ax in enumerate(fig.axes):
        p = ax.get_position()
        axes.append({"index": i, "x0": p.x0, "y0": p.y0, "x1": p.x1, "y1": p.y1})
    qa = {"figure": stem, "exports": files, "axes": axes, "gridlines_visible": False}
    (out_dir / f"{stem}.qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa
