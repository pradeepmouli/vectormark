"""Session-scoped storage for editable drawing versions."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Mapping

from .drawing_trace import TraceResult


class DrawingNotFound(Exception):
    """Raised when a drawing is unavailable to the requesting session."""

    error_code = "DRAWING_NOT_FOUND"


@dataclass(frozen=True)
class DrawingVersion:
    id: str
    parent_id: str | None
    plan: Mapping[str, object] | None
    scene: object | None
    label: str | None


@dataclass
class DrawingState:
    id: str
    trace: TraceResult
    versions: dict[str, DrawingVersion]
    child_counts: dict[str, int]
    last_access: float


class DrawingStore:
    """Keep drawings private to a session until their idle timeout elapses."""

    def __init__(
        self,
        *,
        idle_ttl_seconds: float = 1800,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._idle_ttl_seconds = idle_ttl_seconds
        self._now = now
        self._drawings: dict[object, dict[str, DrawingState]] = {}
        self._lock = RLock()

    def create(self, session: object, trace: TraceResult) -> DrawingState:
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            drawing = DrawingState(
                id=f"drw_{secrets.token_urlsafe(18)}",
                trace=trace,
                versions={
                    "v0": DrawingVersion(
                        id="v0",
                        parent_id=None,
                        plan=None,
                        scene=None,
                        label=None,
                    )
                },
                child_counts={"v0": 0},
                last_access=now,
            )
            self._drawings.setdefault(session, {})[drawing.id] = drawing
            return drawing

    def get(
        self,
        session: object,
        drawing_id: str,
        version_id: str,
    ) -> tuple[DrawingState, DrawingVersion]:
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            drawing = self._drawing_for(session, drawing_id)
            try:
                version = drawing.versions[version_id]
            except KeyError as exc:
                raise DrawingNotFound from exc
            drawing.last_access = now
            return drawing, version

    def append(
        self,
        session: object,
        drawing_id: str,
        base_version: str,
        *,
        plan: Mapping[str, object],
        scene: object,
        label: str | None = None,
    ) -> DrawingVersion:
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            drawing = self._drawing_for(session, drawing_id)
            if base_version not in drawing.versions:
                raise DrawingNotFound

            child_number = drawing.child_counts[base_version]
            version_id = f"{base_version}.{child_number}"
            drawing.child_counts[base_version] = child_number + 1
            version = DrawingVersion(
                id=version_id,
                parent_id=base_version,
                plan=plan,
                scene=scene,
                label=label,
            )
            drawing.versions[version_id] = version
            drawing.child_counts[version_id] = 0
            drawing.last_access = now
            return version

    def _drawing_for(self, session: object, drawing_id: str) -> DrawingState:
        try:
            return self._drawings[session][drawing_id]
        except KeyError as exc:
            raise DrawingNotFound from exc

    def _evict_expired(self, now: float) -> None:
        for session, drawings in list(self._drawings.items()):
            for drawing_id, drawing in list(drawings.items()):
                if now - drawing.last_access >= self._idle_ttl_seconds:
                    del drawings[drawing_id]
            if not drawings:
                del self._drawings[session]
