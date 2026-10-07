"""Externalized user-visible figure text."""
from __future__ import annotations

import json
from pathlib import Path

_RESOURCE = Path(__file__).resolve().parents[1] / "resources" / "labels.json"
LABELS = json.loads(_RESOURCE.read_text(encoding="utf-8"))
