#!/usr/bin/env python3
"""Run the SoraAgent benchmark from the benchmark directory.

Examples::

    python run_benchmark.py
    python run_benchmark.py --suites tools caching --runs 2
    python run_benchmark.py --mode live --provider openai --model gpt-5.4-mini
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sora_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
