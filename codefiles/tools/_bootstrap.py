"""Make the src-layout package importable when a tool is run as a file."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
source = str(SOURCE_ROOT)
if source not in sys.path:
    sys.path.insert(0, source)
