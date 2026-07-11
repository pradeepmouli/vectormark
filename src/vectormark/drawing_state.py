"""Session-scoped storage for immutable, editable drawing versions."""

from __future__ import annotations

import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Callable

import numpy as np

from .drawing_trace import TraceRegion, TraceResult


class DrawingNotFound(Exception):
    """Raised when a drawing is unavailable to the requesting session."""

    error_code = "DRAWING_NOT_FOUND"


@dataclass(frozen=True)
class DrawingVersion:
    """An immutable public snapshot of one retained drawing version."""

    id: str
    parent_id: str | None
    plan: Mapping[str, object] | None
    scene: object | None
    label: str | None


@dataclass(frozen=True)
class DrawingState:
    """An immutable public snapshot of a drawing and its retained versions."""

    id: str
    trace: TraceResult
    versions: Mapping[str, DrawingVersion]


@dataclass
class _StoredDrawing:
    """Mutable bookkeeping that never leaves ``DrawingStore``."""

    id: str
    trace: TraceResult
    versions: dict[str, DrawingVersion]
    child_counts: dict[str, int]
    last_access: float


_IMMUTABLE_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _freeze(value: object, ancestors: set[int] | None = None) -> object:
    """Detach supported containers and reject values without a safe frozen form."""
    snapshot = getattr(value, "__drawing_state_snapshot__", None)
    if callable(snapshot):
        return snapshot()
    if type(value) in _IMMUTABLE_SCALAR_TYPES:
        return value

    if not isinstance(value, Mapping | list | tuple | set | frozenset):
        raise TypeError(f"unsupported value type: {type(value).__qualname__}")

    ancestors = set() if ancestors is None else ancestors
    value_id = id(value)
    if value_id in ancestors:
        raise TypeError("cyclic container values are unsupported")
    ancestors.add(value_id)
    try:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {_freeze(key, ancestors): _freeze(item, ancestors) for key, item in value.items()}
            )
        if isinstance(value, list | tuple):
            return tuple(_freeze(item, ancestors) for item in value)
        return frozenset(_freeze(item, ancestors) for item in value)
    finally:
        ancestors.remove(value_id)


def _freeze_plan(plan: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze(plan)
    assert isinstance(frozen, Mapping)
    return frozen


def _snapshot_trace(trace: TraceResult) -> TraceResult:
    """Detach trace arrays while preserving the trace artifact's public shape."""
    regions = []
    for region in trace.regions:
        mask = np.array(region.mask, copy=True)
        mask.setflags(write=False)
        contours = tuple(np.array(contour, copy=True) for contour in region.contours)
        for contour in contours:
            contour.setflags(write=False)
        regions.append(
            TraceRegion(
                id=region.id,
                source_label=region.source_label,
                color=region.color,
                mask=mask,
                contours=contours,
                trace_path=region.trace_path,
                effective_trace_level=region.effective_trace_level,
            )
        )
    return TraceResult(
        width=trace.width,
        height=trace.height,
        options=trace.options,
        regions=tuple(regions),
        region_map_svg=trace.region_map_svg,
    )


def _validate_label(label: str | None) -> None:
    if type(label) not in (str, type(None)):
        raise TypeError("label must be exactly str or None")


def _public_version(version: DrawingVersion) -> DrawingVersion:
    """Return a detached view so callers can never mutate retained state."""
    return DrawingVersion(
        id=version.id,
        parent_id=version.parent_id,
        plan=_freeze_plan(version.plan) if version.plan is not None else None,
        scene=_freeze(version.scene) if version.scene is not None else None,
        label=version.label,
    )


def _public_drawing(drawing: _StoredDrawing) -> DrawingState:
    versions = {
        version_id: _public_version(version)
        for version_id, version in drawing.versions.items()
    }
    return DrawingState(
        id=drawing.id,
        trace=_snapshot_trace(drawing.trace),
        versions=MappingProxyType(versions),
    )


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
        self._drawings: dict[object, dict[str, _StoredDrawing]] = {}
        self._lock = RLock()

    def create(self, session: object, trace: TraceResult) -> DrawingState:
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            drawing = _StoredDrawing(
                id=f"drw_{secrets.token_urlsafe(18)}",
                trace=_snapshot_trace(trace),
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
            return _public_drawing(drawing)

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
                drawing.versions[version_id]
            except KeyError:
                raise DrawingNotFound from None
            drawing.last_access = now
            public_drawing = _public_drawing(drawing)
            return public_drawing, public_drawing.versions[version_id]

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
                raise DrawingNotFound from None

            _validate_label(label)
            frozen_plan = _freeze_plan(plan)
            frozen_scene = _freeze(scene)
            child_number = drawing.child_counts[base_version]
            version_id = f"{base_version}.{child_number}"
            drawing.child_counts[base_version] = child_number + 1
            version = DrawingVersion(
                id=version_id,
                parent_id=base_version,
                plan=frozen_plan,
                scene=frozen_scene,
                label=label,
            )
            drawing.versions[version_id] = version
            drawing.child_counts[version_id] = 0
            drawing.last_access = now
            return _public_version(version)

    def _drawing_for(self, session: object, drawing_id: str) -> _StoredDrawing:
        try:
            return self._drawings[session][drawing_id]
        except KeyError:
            raise DrawingNotFound from None

    def _evict_expired(self, now: float) -> None:
        for session, drawings in list(self._drawings.items()):
            for drawing_id, drawing in list(drawings.items()):
                if now - drawing.last_access >= self._idle_ttl_seconds:
                    del drawings[drawing_id]
            if not drawings:
                del self._drawings[session]
