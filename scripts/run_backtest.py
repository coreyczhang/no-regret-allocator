#!/usr/bin/env python3
"""Run the full backtest from a source checkout without installing the package.

Adds ``src/`` to the path, then delegates to :func:`nra.cli.main`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nra.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
