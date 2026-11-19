from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _safe_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SORA_MEMORY", "0")
    monkeypatch.setenv("SORA_MEMORY_DREAM_AUTO", "0")
    monkeypatch.setenv("SORA_MEMORY_AUTO_DOWNLOAD", "0")
    monkeypatch.setenv("SORA_IDENTITY", "0")
    monkeypatch.setenv("SORA_CALENDAR_DB", str(tmp_path / "calendar.db"))
    monkeypatch.setenv("SORA_CONFIG", str(tmp_path / "config.json"))
    monkeypatch.setenv("SORA_SECRETS", str(tmp_path / "secrets.env"))
