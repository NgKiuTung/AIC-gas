# Phase7.1 Online Result

This directory archives the Phase7.1 long-horizon implementation used in the
final online combination.

## Final online score

Short component:
- Frozen Phase6-lineage Short
- Accuracy: 90.48%
- generator_1: 87.17%
- generator_all: 93.78%
- Score: 30.9525 / 50

Long component:
- Phase7.1 selected Long
- Accuracy: 84.71%
- generator_1: 82.43%
- generator_all: 87.00%
- Score: 29.1378 / 50

Total:
- 60.0903 / 100

## Phase7.1 Long selection

generator_1:
- Retain reference_58 prediction for all 96 horizons.

generator_all:
- h01-08: xgb_wide_180d_d50_t12
- h09-24: xgb_wide_180d_d50_t12
- h25-48: xgb_wide_180d_d50_t12
- h49-96: xgb_wide_180d_d50_t12

## Frozen online artifact identity

Phase7.1 selected Long CSV SHA-256:

b86beb4dc8e13ccee30d4cf3c98c99404307c2d84cc8d5ad55d4caa1f2532648

Verification:
- failed jobs: 0
- timed-out jobs: 0
- replay max error: 0
- test targets used: false
