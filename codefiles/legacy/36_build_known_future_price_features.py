"""Add deterministic future tariff-path features for each forecast horizon."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_cleaning_enhanced.pkl"
CATALOG_PATH = ROOT / "results" / "features" / "feature_catalog_cleaning_enhanced.csv"
PRICE_PATH = ROOT / "dataset" / "初赛-数据集" / "price.xlsx"
OUTPUT_PATH = ROOT / "results" / "features" / "train_supervised_features_price_augmented.pkl"
OUTPUT_CATALOG = ROOT / "results" / "features" / "feature_catalog_price_augmented.csv"
AUDIT_PATH = ROOT / "results" / "features" / "known_future_price_feature_audit.json"
HORIZONS = tuple(range(1, 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def price_lookup() -> dict[tuple[int, int], float]:
    table = pd.read_excel(PRICE_PATH)
    lookup: dict[tuple[int, int], float] = {}
    for row_index, row in table.iterrows():
        for month in range(1, 13):
            lookup[(month, row_index)] = float(row[f"{month}月"])
    return lookup


def lookup_series(datetimes: pd.Series, lookup: dict[tuple[int, int], float]) -> np.ndarray:
    slot = datetimes.dt.hour.to_numpy() * 2 + (datetimes.dt.minute.to_numpy() >= 30).astype(int)
    month = datetimes.dt.month.to_numpy()
    return np.array([lookup[(int(m), int(s))] for m, s in zip(month, slot)], dtype=np.float32)


def main() -> None:
    data = pd.read_pickle(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
    lookup = price_lookup()
    current = lookup_series(data["datetime"], lookup)
    future_prices = np.column_stack(
        [lookup_series(data["datetime"] + pd.Timedelta(minutes=15 * h), lookup) for h in HORIZONS]
    )
    added: dict[str, np.ndarray] = {}
    for index, horizon in enumerate(HORIZONS):
        price = future_prices[:, index]
        path = future_prices[:, : index + 1]
        previous_path = np.column_stack([current, path[:, :-1]])
        changes = np.abs(path - previous_path) > 1e-9
        added[f"feat_future_price_h{horizon}"] = price
        added[f"feat_future_price_delta_h{horizon}"] = price - current
        added[f"feat_future_price_up_h{horizon}"] = (price > current).astype(np.float32)
        added[f"feat_future_price_down_h{horizon}"] = (price < current).astype(np.float32)
        added[f"feat_future_price_path_mean_h{horizon}"] = path.mean(axis=1)
        added[f"feat_future_price_path_min_h{horizon}"] = path.min(axis=1)
        added[f"feat_future_price_path_max_h{horizon}"] = path.max(axis=1)
        added[f"feat_future_price_path_range_h{horizon}"] = path.max(axis=1) - path.min(axis=1)
        added[f"feat_future_price_change_count_h{horizon}"] = changes.sum(axis=1).astype(np.float32)
    all_changes = np.abs(future_prices - np.column_stack([current, future_prices[:, :-1]])) > 1e-9
    first_change = np.where(all_changes.any(axis=1), all_changes.argmax(axis=1) + 1, 9).astype(np.float32)
    first_change_price = np.array(
        [future_prices[row, int(step) - 1] if step <= 8 else current[row] for row, step in enumerate(first_change)],
        dtype=np.float32,
    )
    added["feat_future_price_steps_to_next_change"] = first_change
    added["feat_future_price_next_change_delta"] = first_change_price - current
    added_frame = pd.DataFrame(added, index=data.index)
    augmented = pd.concat([data, added_frame], axis=1)
    if augmented[list(added)].isna().any().any() or not np.isfinite(augmented[list(added)].to_numpy()).all():
        raise ValueError("Known future price features contain invalid values")
    catalog = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    new_catalog = pd.DataFrame({"feature": list(added), "group": "known_future_price"})
    catalog = pd.concat([catalog, new_catalog], ignore_index=True)
    augmented.to_pickle(OUTPUT_PATH)
    catalog.to_csv(OUTPUT_CATALOG, index=False, encoding="utf-8-sig")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "training features plus deterministic tariff schedule from price.xlsx",
        "external_scoring_data_accessed": False,
        "future_observations_used": False,
        "known_future_input": "tariff by month and half-hour",
        "rows": len(augmented), "added_feature_count": len(added),
        "total_feature_count": len(catalog),
        "rows_with_price_change_within_120m": int((first_change <= 8).sum()),
        "output": str(OUTPUT_PATH.relative_to(ROOT)), "output_sha256": sha256(OUTPUT_PATH),
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
