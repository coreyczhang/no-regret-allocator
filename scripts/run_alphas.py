#!/usr/bin/env python3
"""Run the alphas and the allocator capstone from a source checkout.

Adds ``src/`` to the path, then delegates to :func:`nra.alphas_cli.main`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nra.alphas_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
