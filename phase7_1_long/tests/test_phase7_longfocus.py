
from pathlib import Path
import json
import numpy as np
import pandas as pd

from gasauto.phase7 import _read_short_source, _component_gate, _freeze_missing_short, Phase7Research
from gasauto.protocol import long_component_scores, read_plan, LONG_BANDS
from gasauto.trees import frozen_day
from gasauto.cli import parser
from gasauto.delivery import task_frame, validate_long_file
from gasbench.common import ROOT


def test_phase7_short_source_reference_shape():
    frame = _read_short_source(ROOT / "reference/reference_58_results.zip")
    assert frame.shape == (960, 17)
    assert frame.columns[0] == "datetime"


def test_phase7_component_gate_requires_wins():
    rule = {"min_mean_gain": 0.0001, "max_fold_regression": 0.01,
            "max_last_regression": 0.01, "min_wins": 2}
    good = _component_gate([0.09, 0.10, 0.11], [0.10, 0.11, 0.12], rule)
    assert good["accepted"]
    bad = _component_gate([0.09, 0.12, 0.13], [0.10, 0.11, 0.12], rule)
    assert not bad["accepted"]
    assert "wins" in bad["reason"]


def test_phase7_long_component_scores_partition():
    truth = np.ones((2, 96, 2), dtype=float) * 100
    pred = truth.copy()
    pred[:, 48:, 1] = 110
    scores = long_component_scores(truth, pred)
    assert scores["g1__h01_08"] == 0
    assert abs(scores["gall__h49_96"] - 0.1) < 1e-12
    assert set(scores) == {f"{t}__{b}" for t in ("g1", "gall") for b in LONG_BANDS}


def test_phase7_config_is_long_only():
    cfg = read_plan(ROOT / "configs/phase7_a10_3h.json")
    assert cfg["mode"] == "phase7_long_component_focus"
    assert cfg["candidates"]
    assert all(c["tasks"] == ["long"] for c in cfg["candidates"])
    assert any(c["kind"] == "proxy_wide" for c in cfg["candidates"])
    assert any(c["family"] == "lightgbm" and c["kind"] == "direct" for c in cfg["candidates"])
    assert any(c["family"] == "mlp" and c["kind"] == "sequence" for c in cfg["candidates"])


def test_frozen_day_does_not_read_after_history():
    idx = pd.date_range("2025-01-01", periods=96*20, freq="15min")
    history = pd.DataFrame({"generator_1": np.arange(len(idx), dtype=float),
                            "generator_all": np.arange(len(idx), dtype=float)+100}, index=idx)
    cutoff = pd.Timestamp("2025-01-15")
    history = history.loc[history.index < cutoff]
    contract = {"protocol": {"block_start_offset_minutes": 0, "blocks": 96, "block_minutes": 15}}
    origins = pd.DatetimeIndex([cutoff, cutoff + pd.Timedelta(days=3)])
    out = frozen_day(history, origins, contract, 96)
    assert out.shape == (2, 96, 2)
    assert np.isfinite(out).all()


def test_phase7_long_only_cli_is_explicit():
    args = parser().parse_args([
        "research", "--cache", "cache/main", "--run", "runs/long", "--long-only"
    ])
    assert args.long_only
    assert args.short_source is None


def test_phase7_missing_short_is_recorded_not_fabricated(tmp_path):
    meta = _freeze_missing_short(tmp_path)
    assert meta["status"] == "unavailable"
    assert not (tmp_path / "frozen_short/s_result.csv").exists()
    saved = json.loads((tmp_path / "frozen_short/source.json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] is None


def test_long_only_validation_never_claims_submission_pair(tmp_path):
    index = pd.date_range("2025-10-01", periods=960, freq="15min")
    values = np.ones((960, 96, 2), dtype=float)
    values[:, :, 1] = 2.0
    path = tmp_path / "l_result.csv"
    task_frame(index, values, 96).to_csv(path, index=False)
    report = validate_long_file(path, index)
    assert report["validation"] == "passed"
    assert not report["submission_pair_ready"]


def test_phase7_long_only_export_has_no_fake_short(tmp_path):
    _freeze_missing_short(tmp_path)
    engine = object.__new__(Phase7Research)
    engine.run = tmp_path
    engine.protocol = {
        "test": {
            "name": "test",
            "start": "2025-10-01 00:00:00",
            "end": "2025-10-10 23:45:00",
        }
    }
    long_pred = np.ones((960, 96, 2), dtype=float)
    long_pred[:, :, 1] = 2.0
    engine._export(long_pred, long_pred, {"test": "long-only"})
    selected = tmp_path / "candidate_results/selected"
    assert (selected / "l_result.csv").exists()
    assert not (selected / "s_result.csv").exists()
    assert not (selected / "results_only.zip").exists()
    assert (selected / "long_only_results.zip").exists()
    report = json.loads((selected / "validation.json").read_text(encoding="utf-8"))
    assert not report["submission_pair_ready"]
