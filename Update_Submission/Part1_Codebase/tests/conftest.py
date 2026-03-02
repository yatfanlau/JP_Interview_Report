"""Pytest configuration shared by all tests.

This file ensures the local ``src/`` package is importable in test runs and
forces a headless Matplotlib backend so plotting code can be exercised in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Ensure local package imports resolve without requiring installation.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Use writable paths and a non-interactive backend for matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
