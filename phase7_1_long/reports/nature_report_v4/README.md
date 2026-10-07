# AIC-Gas Phase7.1 Nature-Skills Figure Patch v4

v4 is the post-review correction of v3. It keeps the Nature-Skills visual contract but fixes semantic and visual defects found in the rendered v3 gallery.

## Why v4 exists

The v3 render was materially better than the original dashboard-style attempt, but visual inspection exposed several problems that block report delivery:

- the pipeline schematic incorrectly implied that frozen Short came after Phase7.1 Long;
- the feature heatmap inferred families from names and produced an uninformative `other families` block;
- the polar panel had label collisions and weak information density;
- feature coverage was parsed from the wrong column semantics;
- the final selection strip failed to read `screen_picks` and displayed `generator_all` as reference;
- numeric-profile `missing` counts were mislabeled as missing ratios;
- the audit panel reduced to one action label and did not show where the 1,004 isolation events occurred.

v4 fixes those issues at the data-contract level rather than only restyling the plots.

## Figure set

1. `figureA_pipeline_and_score`
   - parallel Long-research and frozen-Short lanes;
   - explicit merge into the final 60.0903 submission;
   - processed-data scale and online contribution support panels.
2. `figureB_feature_architecture`
   - physical-domain × transformation-family heatmap from exact dictionary `source`/`operator` metadata;
   - transform-family donut composition;
   - 276-tile provenance map: 270 causal, 6 known-before-origin, 0 target-derived.
3. `figureC_model_selection_evidence`
   - screening gain heatmap;
   - confirmation stability view;
   - authoritative final assignment parsed from `long.screen_picks`.
4. `figureD_online_performance`
   - discrete online-score progression with deltas;
   - Short/Long contribution dumbbell;
   - Short/Long accuracy dumbbell.
5. `figureE_data_quality`
   - train/test feature-availability distribution;
   - raw-field missingness percentages using inventory row denominators;
   - isolated-value concentration by field on a log scale.

Each figure exports editable PDF/SVG, TIFF at 600 dpi, PNG at 300 dpi, and QA JSON.

## Run on DSW

```bash
PATCH=/mnt/workspace/AIC-Gas-Phase6-Rebuild/tmp/nature-viz-v4/aic_gas_nature_report_patch_v4
REPO=/mnt/workspace/AIC-Gas-Phase6-Rebuild/AIC-gas
OUT=/mnt/workspace/AIC-Gas-Phase6-Rebuild/nature-viz-v4

bash "$PATCH/run_all.sh" "$REPO" "$REPO/phase7_1_long" "$OUT"
```

Review:

```text
/mnt/workspace/AIC-Gas-Phase6-Rebuild/nature-viz-v4/REVIEW_GALLERY.html
```

Final technical-report assets should use PDF/SVG/TIFF, not the review-gallery PNGs.
