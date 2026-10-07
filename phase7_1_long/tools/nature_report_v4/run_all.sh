#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-.}"
PHASE7="${2:-$REPO/phase7_1_long}"
OUT="${3:-$REPO/reports/phase7_1_nature_v4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$HERE/tools/nature_report_v4/run.py" \
  --repo-root "$REPO" \
  --phase7-dir "$PHASE7" \
  --out "$OUT"

python3 "$HERE/tools/nature_report_v4/build_gallery.py" \
  --figures "$OUT" \
  --out "$OUT/REVIEW_GALLERY.html"

echo "FIGURES=$OUT"
echo "GALLERY=$OUT/REVIEW_GALLERY.html"
