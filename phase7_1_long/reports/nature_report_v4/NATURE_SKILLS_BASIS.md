# Nature-Skills implementation basis

This patch follows the public `Yuan1z0825/nature-skills` `skills/nature-figure` guidance reviewed at commit:

`7d5f160ebfe8033375b4afef7c5911aa4203c983`

Applied principles:

- claim-first figure contract: conclusion → evidence chain → archetype → export contract;
- narrative figure pages rather than dashboard-style panels;
- one dominant evidence role with subordinate support panels;
- small bold lowercase panel labels;
- white background and no background grid by default;
- restrained semantic palette reused across figures;
- green/red reserved for directional gain/drop evidence;
- direct labels where they reduce legend travel;
- asymmetric layouts when evidence importance differs;
- heatmap, donut composition, tile/waffle provenance, interval diagnostics, selection strip, connected-dot progression, dumbbells, distribution strips and log-scale audit diagnostics selected by evidence role;
- editable SVG/PDF text plus 600-dpi TIFF export.

v4 also applies the skill's data-integrity stance: figure semantics are derived from the archived schema rather than guessed from the first numeric column. In particular, `coverage.csv` is interpreted as missing fractions, `numeric_profile.csv` missingness is normalized with `inventory.csv` row counts, and the final model allocation is read from `selection.long.screen_picks`.

The repository is Apache-2.0 licensed. This patch is an original implementation of its figure-design principles rather than a copy of its example figures.
