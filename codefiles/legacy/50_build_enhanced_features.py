"""Build enhanced feature set with future prices and interactions.

This script extends the existing feature pipeline with:
1. Future known price features (80 features)
2. Interaction features (22 features)

Expected impact: 3-7% MAPE improvement
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codefiles" / "src"))

from gas_power.features.enhanced_interactions import build_enhanced_features

# Input/Output paths
INPUT_FEATURES = ROOT / "results" / "features" / "train_supervised_features.pkl"
INPUT_CATALOG = ROOT / "results" / "features" / "feature_catalog.csv"
PRICE_PATH = ROOT / "dataset" / "初赛-数据集" / "price.xlsx"

OUTPUT_FEATURES = ROOT / "results" / "features" / "train_supervised_features_enhanced.pkl"
OUTPUT_CATALOG = ROOT / "results" / "features" / "feature_catalog_enhanced.csv"
AUDIT_PATH = ROOT / "results" / "features" / "enhanced_features_audit.json"


def sha256(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_price_lookup() -> dict[tuple[int, int], float]:
    """Load electricity price lookup table from Excel.

    Returns:
        Dictionary mapping (month, half_hour_slot) -> price
        where half_hour_slot is 0-47 (0=00:00-00:30, 1=00:30-01:00, etc.)
    """
    print("Loading price lookup table...")
    table = pd.read_excel(PRICE_PATH)

    lookup: dict[tuple[int, int], float] = {}
    for row_index, row in table.iterrows():
        for month in range(1, 13):
            lookup[(month, row_index)] = float(row[f"{month}月"])

    print(f"  Loaded {len(lookup)} price entries")
    print(f"  Months: 1-12, Slots: 0-{len(table)-1}")
    print(f"  Price range: {min(lookup.values()):.3f} - {max(lookup.values()):.3f}")

    return lookup


def main() -> None:
    """Main pipeline."""
    print("=" * 80)
    print("Building Enhanced Features (Future Prices + Interactions)")
    print("=" * 80)

    start_time = time.perf_counter()

    # Load existing features
    print(f"\nLoading existing features from {INPUT_FEATURES.name}...")
    data = pd.read_pickle(INPUT_FEATURES)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")

    print(f"  Rows: {len(data):,}")
    print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"  Existing features: {data.shape[1]}")

    # Load existing catalog
    catalog = pd.read_csv(INPUT_CATALOG, encoding="utf-8-sig")
    print(f"  Features in catalog: {len(catalog)}")

    # Check required base features
    required_features = [
        "datetime", "feat_known_price", "feat_price_level",
        "blast_furnace_gas_holder_2", "feat_p50_current", "feat_p120_current",
        "feat_generator_all_filled", "generator_use_blast_furnace_gas",
        "feat_blast_furnace_observed_sum", "feat_month", "feat_hour", "feat_is_weekend"
    ]

    missing = [f for f in required_features if f not in data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    print("\n  All required base features present")

    # Load price lookup
    price_lookup = load_price_lookup()

    # Build enhanced features
    print("\nBuilding enhanced features...")
    print("  Step 1: Adding future price features (80 features)...")
    print("  Step 2: Adding interaction features (22 features)...")

    enhanced = build_enhanced_features(data, price_lookup)

    new_features = enhanced.shape[1] - data.shape[1]
    print(f"\n  Added {new_features} new features")
    print(f"  Total features: {enhanced.shape[1]}")

    # Identify new feature names
    new_feature_names = [col for col in enhanced.columns if col not in data.columns]

    # Categorize new features
    future_price_features = [f for f in new_feature_names if "future_price" in f]
    interaction_features = [f for f in new_feature_names if "interact" in f]

    print(f"\n  Breakdown:")
    print(f"    - Future price features: {len(future_price_features)}")
    print(f"    - Interaction features: {len(interaction_features)}")

    # Validate: no NaN or inf
    print("\nValidating new features...")
    if enhanced[new_feature_names].isna().any().any():
        raise ValueError("NaN values detected in new features!")

    import numpy as np
    if not np.isfinite(enhanced[new_feature_names].to_numpy()).all():
        raise ValueError("Non-finite values detected in new features!")

    print("  All features valid (no NaN/inf)")

    # Update catalog
    print("\nUpdating feature catalog...")
    new_catalog_entries = []

    for feature in future_price_features:
        new_catalog_entries.append({
            "feature": feature,
            "group": "future_price"
        })

    for feature in interaction_features:
        new_catalog_entries.append({
            "feature": feature,
            "group": "interaction"
        })

    new_catalog_df = pd.DataFrame(new_catalog_entries)
    updated_catalog = pd.concat([catalog, new_catalog_df], ignore_index=True)

    print(f"  Catalog entries: {len(catalog)} -> {len(updated_catalog)}")

    # Save outputs
    print("\nSaving outputs...")
    enhanced.to_pickle(OUTPUT_FEATURES)
    print(f"  Features saved to: {OUTPUT_FEATURES.name}")

    updated_catalog.to_csv(OUTPUT_CATALOG, index=False, encoding="utf-8-sig")
    print(f"  Catalog saved to: {OUTPUT_CATALOG.name}")

    # Create audit log
    elapsed = time.perf_counter() - start_time

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "input_features": str(INPUT_FEATURES.relative_to(ROOT)),
        "input_features_sha256": sha256(INPUT_FEATURES),
        "output_features": str(OUTPUT_FEATURES.relative_to(ROOT)),
        "output_features_sha256": sha256(OUTPUT_FEATURES),
        "rows": len(enhanced),
        "datetime_range": {
            "min": data["datetime"].min().isoformat(),
            "max": data["datetime"].max().isoformat(),
        },
        "features": {
            "original": data.shape[1],
            "added": new_features,
            "total": enhanced.shape[1],
            "future_price_count": len(future_price_features),
            "interaction_count": len(interaction_features),
        },
        "validation": {
            "no_nan": True,
            "no_inf": True,
            "all_finite": True,
        },
        "performance": {
            "elapsed_seconds": elapsed,
            "rows_per_second": len(enhanced) / elapsed,
        },
        "expected_impact": "3-7% MAPE improvement",
    }

    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  Audit log saved to: {AUDIT_PATH.name}")

    # Summary
    print("\n" + "=" * 80)
    print("SUCCESS: Enhanced features built successfully!")
    print("=" * 80)
    print(f"\nProcessing time: {elapsed:.2f} seconds")
    print(f"Throughput: {len(enhanced) / elapsed:,.0f} rows/second")

    memory_mb = enhanced.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Review the enhanced features:")
    print(f"   - {OUTPUT_FEATURES}")
    print(f"   - {OUTPUT_CATALOG}")
    print("\n2. Retrain the model using enhanced features:")
    print("   - Modify 20_train_final_forecaster.py to use the new feature file")
    print("   - Run training and compare MAPE")
    print("\n3. Expected improvement: 3-7% MAPE reduction")
    print("   - Current MAPE: Check validation results")
    print("   - Target MAPE: Current * 0.93-0.97")
    print("\n4. If results are good, update the production pipeline")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Feature Statistics Sample:")
    print("=" * 80)

    # Sample future price features
    print("\nFuture Price Features (sample):")
    for h in [1, 4, 8]:
        col = f"feat_future_price_h{h}"
        if col in enhanced.columns:
            print(f"  {col}: mean={enhanced[col].mean():.3f}, std={enhanced[col].std():.3f}")

    # Sample interaction features
    print("\nInteraction Features (sample):")
    key_interactions = [
        "feat_interact_price_holder",
        "feat_interact_price_p50",
        "feat_interact_weekend_hour",
    ]
    for col in key_interactions:
        if col in enhanced.columns:
            print(f"  {col}: mean={enhanced[col].mean():.3f}, std={enhanced[col].std():.3f}")

    print("\n" + json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
