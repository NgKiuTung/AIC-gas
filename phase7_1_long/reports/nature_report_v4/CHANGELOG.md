# Changelog

## v4 — post-render correctness and composition pass

- Corrected Figure A provenance: Phase6 Short and Phase7.1 Long are parallel inputs to the final submission, not a serial dependency.
- Replaced inferred feature-name families with exact `source` and `operator` metadata from `dictionary.csv`.
- Replaced the awkward polar composition with a directly labelled donut and a 276-feature provenance tile map.
- Removed the semantically incorrect coverage-ECDF parser from Figure B.
- Fixed Figure C final selection by reading `selection.long.screen_picks`; `generator_all` now correctly resolves to XGB t12 for all four bands while `generator_1` remains reference.
- Improved confirmation stability encoding and candidate-row emphasis.
- Added explicit online score deltas to Figure D.
- Corrected Figure E missingness semantics: raw missing counts are divided by source row counts from `inventory.csv` before plotting percentages.
- Reworked audit evidence to show concentration by field rather than a single repeated action label.
- Externalized user-visible labels to `resources/labels.json`.
- Added 15 regression tests for semantic mappings, selection parsing, and missingness normalization.

## v3 — Nature-Skills redesign

- Removed default chart grids and oversized plot titles.
- Introduced small panel labels, semantic low-saturation palette, multi-panel evidence architecture, and PDF/SVG/TIFF exports.
