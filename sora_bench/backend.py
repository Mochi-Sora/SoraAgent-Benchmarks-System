"""Deterministic in-memory stand-in for CortexDB.

The real memory subsystem talks to an external sidecar.  Benchmarks must not
download or spawn binaries, so this backend implements the small protocol the
recall and dream pipelines need, with call counters and optional latency for
efficiency accounting.
"""

from __future__ import annotations

import copy
import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "need",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)


def _tokens(text: str) -> set:
    value = str(text or "").lower()
    return {token for token in _TOKEN_RE.findall(value) if token not in _STOPWORDS} | set(
        _CJK_RE.findall(value)
    )


class FakeMemoryBackend:
    """A small, observable long-term memory backend."""

    def __init__(self, *, latency_ms: float = 0.0) -> None:
        self.latency_ms = max(0.0, float(latency_ms))
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.calls: Dict[str, int] = {}
        self.fail_save = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _maybe_sleep(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def stats(self) -> Dict[str, Any]:
        return {
            "memories": len(self.memories),
            "calls": dict(sorted(self.calls.items())),
            "total_calls": self.total_calls(),
            "bytes": sum(
                len(str(record.get("content") or "").encode("utf-8"))
                for record in self.memories.values()
            ),
        }

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.memories)

    def restore(self, snapshot: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            self.memories = copy.deepcopy(snapshot)

    # ------------------------------------------------------------------
    # MemoryBackend protocol
    # ------------------------------------------------------------------
    def health(self) -> bool:
        self._count("health")
        self._maybe_sleep()
        return True

    def save_memory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._count("save_memory")
        self._maybe_sleep()
        if self.fail_save:
            raise RuntimeError("injected save failure")
        memory_id = str(payload.get("memory_id") or "")
        if not memory_id:
            raise ValueError("memory_id is required")
        with self._lock:
            existing = self.memories.get(memory_id, {})
            metadata = dict(existing.get("metadata") or {})
            metadata.update(dict(payload.get("metadata") or {}))
            record = {
                "id": memory_id,
                "content": str(payload.get("content") or ""),
                "metadata": metadata,
                "importance": float(payload.get("importance", existing.get("importance", 0.5))),
                "created_at": str(
                    payload.get("created_at")
                    or existing.get("created_at")
                    or datetime.now(timezone.utc).isoformat()
                ),
                "user_id": str(payload.get("user_id") or "sora"),
                "session_id": str(payload.get("session_id") or "default"),
                "role": str(payload.get("role") or ""),
                "scope": str(payload.get("scope") or "user"),
            }
            for key in ("expires_at", "supersedes", "entities", "relations"):
                if key in payload:
                    record[key] = copy.deepcopy(payload[key])
            self.memories[memory_id] = record
            return {"memory": copy.deepcopy(record)}

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        self._count("get_memory")
        self._maybe_sleep()
        with self._lock:
            record = self.memories.get(str(memory_id))
            return copy.deepcopy(record) if record is not None else None

    def list_memories(self, limit: int = 500) -> List[Dict[str, Any]]:
        self._count("list_memories")
        self._maybe_sleep()
        with self._lock:
            records = [copy.deepcopy(self.memories[key]) for key in sorted(self.memories)]
        return records[: max(1, int(limit))]

    def update_memory(self, memory_id: str, **fields: Any) -> Dict[str, Any]:
        self._count("update_memory")
        self._maybe_sleep()
        with self._lock:
            record = self.memories.get(str(memory_id))
            if record is None:
                return {}
            for key, value in fields.items():
                if key == "metadata":
                    metadata = dict(record.get("metadata") or {})
                    metadata.update(dict(value or {}))
                    record["metadata"] = metadata
                else:
                    record[key] = value
            return {"memory": copy.deepcopy(record)}

    def delete_memory(self, memory_id: str) -> None:
        self._count("delete_memory")
        self._maybe_sleep()
        with self._lock:
            self.memories.pop(str(memory_id), None)

    def recall(self, query: str, top_k: int = 8, **options: Any) -> Dict[str, Any]:
        self._count("recall")
        self._maybe_sleep()
        terms = _tokens(query)
        if not terms:
            return {"memories": []}
        scored: List[tuple] = []
        with self._lock:
            for memory_id, record in self.memories.items():
                content = str(record.get("content") or "")
                overlap = terms & _tokens(content)
                if not overlap:
                    continue
                # Cosine-ish lexical score with a small recency nudge.
                score = len(overlap) / math.sqrt(max(1, len(terms)))
                score += 0.01 * float(record.get("importance") or 0.0)
                scored.append((score, memory_id, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = max(1, min(int(top_k), 50))
        return {
            "memories": [
                {
                    "memory": copy.deepcopy(record),
                    "score": round(score, 6),
                }
                for score, _memory_id, record in scored[:limit]
            ]
        }

    def rules_apply(self, **payload: Any) -> Dict[str, Any]:
        self._count("rules_apply")
        self._maybe_sleep()
        return {"derived": 0, "dry_run": bool(payload.get("dry_run"))}

    def sweep_expired(self) -> Dict[str, Any]:
        self._count("sweep_expired")
        self._maybe_sleep()
        return {"swept": 0}

    def close(self) -> None:
        return None
