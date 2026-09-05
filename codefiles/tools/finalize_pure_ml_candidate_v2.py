"""Archive paired retrospective diagnostics without opening scoring data."""

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/experiments/pure_ml_candidate_v2/20260905T071738Z"
BASE = ROOT / "results/experiments/state_guarded_feature_screen/20260905T070621Z"


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    candidate = rows(RUN / "predictions.csv.gz")
    baseline = [r for r in rows(BASE / "predictions.csv.gz") if r["variant"] == "baseline"]
    keys = ("origin", "window", "target", "horizon")
    reference = {tuple(r[k] for k in keys): r for r in baseline}
    assert len(reference) == len(baseline) == len(candidate)
    assert len({tuple(r[k] for k in keys) for r in candidate}) == len(candidate)
    groups = defaultdict(list)
    for row in candidate:
        old = reference[tuple(row[k] for k in keys)]
        assert abs(float(row["actual"]) - float(old["actual"])) < 1e-8
        assert float(row["actual"]) > 0, "Official MAPE zero-label policy needs clarification"
        pair = (float(old["ape"]) * 100, float(row["ape"]) * 100)
        for group in ("all", row["window"], row["target"], "horizon_" + row["horizon"]):
            groups[group].append(pair)
    comparison = {}
    for group, pairs in groups.items():
        old = sum(p[0] for p in pairs) / len(pairs)
        new = sum(p[1] for p in pairs) / len(pairs)
        comparison[group] = dict(n=len(pairs), baseline_mape_pct=old, candidate_mape_pct=new, gain_pp=old-new)
    summary = dict(
        comparison=comparison,
        identical_origin_target_horizon_and_truth=True,
        zero_validation_labels=0,
        promoted=False,
        reason="Mean improved but latest window regressed; reused dates, different model/feature recipes, not a single-factor ablation.",
    )
    (RUN / "paired_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sources = [
        ROOT / "codefiles/tools/build_pure_ml_candidate_v2.py",
        Path(__file__),
        ROOT / "codefiles/src/gas_power/features/spectral.py",
        ROOT / "codefiles/src/gas_power/data/causal_preprocessing.py",
        ROOT / "codefiles/src/gas_power/submission/input_schema.py",
    ]
    artifacts = sorted(p for p in RUN.rglob("*") if p.is_file() and p.name != "artifact_hashes.json")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts + sources}
    (RUN / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
