"""Small runtime helpers shared by the suites."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sora_agent.approval import AutoApprovalGate
from sora_agent.identity import IdentityConfig, IdentityStore


class FixedClock:
    """A callable clock that always returns the same time."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment = self.moment + timedelta(seconds=seconds)


class SteppingClock:
    """Advance by one second per provider turn, deterministically.

    This models the real wall clock closely enough to expose cache-breaking
    prompt inserts without depending on how fast the machine runs.
    """

    def __init__(
        self,
        start: Optional[datetime] = None,
        step_seconds: float = 1.0,
    ) -> None:
        self.moment = start or datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)
        self.step_seconds = float(step_seconds)
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            current = self.moment
            self.moment = self.moment + timedelta(seconds=self.step_seconds)
            return current


def fixed_clock() -> datetime:
    return datetime(2026, 11, 12, 9, 30, 0, tzinfo=timezone.utc)


def build_identity(root: Path) -> IdentityStore:
    """Create a small valid USER.md/SOUL.md pair under ``root``."""

    root = Path(root)
    store = IdentityStore(IdentityConfig(root=root))
    user = store.apply(
        "user",
        "- Name: Marco Tan\n"
        "- Role: Operations lead\n"
        "- Timezone: Asia/Macau\n"
        "- Preference: concise status updates",
    )
    soul = store.apply(
        "soul",
        "- Voice: calm and precise\n"
        "- Values: safety first\n"
        "- Boundaries: ask before irreversible actions",
    )
    if user.status not in {"accepted", "unchanged"}:
        raise RuntimeError("benchmark identity USER.md was rejected")
    if soul.status not in {"accepted", "unchanged"}:
        raise RuntimeError("benchmark identity SOUL.md was rejected")
    return store


def auto_approval() -> AutoApprovalGate:
    return AutoApprovalGate()


class NoopBackup:
    """Backup stand-in so dream benchmarks need no zstd binary or database."""

    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}
        self.bytes = 0
        self.started = time.perf_counter()

    def _count(self, name: str, source: Optional[Path] = None) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if source is not None:
            try:
                self.bytes += Path(source).stat().st_size
            except OSError:
                pass

    def snapshot_daily(self, source: Path, day: Any = None) -> Dict[str, Any]:
        self._count("snapshot_daily", source)
        return {
            "file": "noop-daily-{}.zst".format(getattr(day, "isoformat", lambda: day)()),
            "bytes": 0,
        }

    def snapshot_specialist_daily(
        self, agent_id: str, source: Path, day: Any = None
    ) -> Dict[str, Any]:
        self._count("snapshot_specialist_daily", source)
        return {
            "file": "noop-specialist-{}-{}.zst".format(
                agent_id, getattr(day, "isoformat", lambda: day)()
            ),
            "bytes": 0,
        }

    def snapshot(
        self, source: Path, kind: str, label: str, **kwargs: Any
    ) -> Dict[str, Any]:
        self._count("snapshot", source)
        return {"file": "noop-{}-{}.zst".format(kind, label), "bytes": 0}

    def snapshot_active(self, source: Optional[Path] = None) -> Dict[str, Any]:
        self._count("snapshot_active", source)
        return {"file": "noop-active.zst", "bytes": 0}

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": dict(sorted(self.calls.items())),
            "total_calls": sum(self.calls.values()),
            "bytes": self.bytes,
        }


class CountingWake:
    """Deterministic wake-model stand-in that counts calls and characters."""

    def __init__(self) -> None:
        self.calls = 0
        self.chars = 0

    def __call__(self, report: Dict[str, Any]) -> str:
        self.calls += 1
        text = "Wake review: {} conflict(s) need human attention.".format(
            len(report.get("conflicts") or [])
        )
        self.chars += len(text)
        return text

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.calls, "chars": self.chars}
