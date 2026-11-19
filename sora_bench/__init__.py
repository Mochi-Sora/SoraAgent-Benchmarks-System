"""SoraAgent benchmark harness.

The benchmark lives next to the SoraAgent checkout in this workspace.  Importing
the package puts the sibling ``SoraAgent`` directory on ``sys.path`` so
``python -m sora_bench`` works from the benchmark directory without installing
anything.  An installed ``sora-agent`` package is used as-is when present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent
_CHECKOUT = _BENCH_ROOT.parent.parent / "SoraAgent"

if _CHECKOUT.is_dir() and str(_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT))

__all__ = ["__version__"]
__version__ = "0.1.0"
