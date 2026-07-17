"""Session-scoped storage for immutable, editable drawing versions."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Callable

import numpy as np

from .candidate import FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .drawing_trace import GeometryTraceRegion, TraceRegion, TraceResult
from .fit import Shape
from .optimizer.vector_region import VectorRegion


class DrawingNotFound(Exception):
    """Raised when a live drawing ID or one of its versions is unavailable."""

    error_code = "DRAWING_NOT_FOUND"


@dataclass(frozen=True)
class DrawingVersion:
    """An immutable public snapshot of one retained drawing version."""

    id: str
    parent_id: str | None
    plan: Mapping[str, object] | None
    regions: tuple[VectorRegion, ...] | None
    label: str | None
    # A geometry-guide retrace owns a different command provenance from the
    # original source trace.  Keep it on the version that introduced it so
    # later path operations validate against the correct command IDs.
    trace: TraceResult | None = field(default=None, compare=False, repr=False)
    geometry_guide_rgb: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class DrawingState:
    """An immutable public snapshot of a drawing and its retained versions."""

    id: str
    trace: TraceResult
    versions: Mapping[str, DrawingVersion]
    source_rgb: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass
class _StoredDrawing:
    """Mutable bookkeeping that never leaves ``DrawingStore``."""

    id: str
    trace: TraceResult
    source_rgb: np.ndarray | None
    versions: dict[str, DrawingVersion]
    child_counts: dict[str, int]
    last_access: float


_IMMUTABLE_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)
_VERSION_ID = re.compile(r"v\d+(?:\.\d+)*$")


def _freeze(value: object, ancestors: set[int] | None = None) -> object:
    """Detach supported containers and reject values without a safe frozen form."""
    snapshot = getattr(value, "__drawing_state_snapshot__", None)
    if callable(snapshot):
        return snapshot()
    if isinstance(value, np.generic):
        return value.item()
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


def _copy_shape(shape: Shape | None) -> Shape | None:
    return None if shape is None else Shape(shape.kind, _freeze(shape.params))


def _copy_fill(fill: object) -> object:
    if isinstance(fill, FlatFill):
        return FlatFill(fill.hex)
    if isinstance(fill, LinearGradientFill):
        return LinearGradientFill(dict(fill.geometry), list(fill.stops))
    if isinstance(fill, RadialGradientFill):
        return RadialGradientFill(dict(fill.geometry), list(fill.stops))
    if isinstance(fill, RasterFill):
        return RasterFill(dict(fill.geometry), fill.png_b64)
    return None


def _snapshot_region(region: VectorRegion) -> VectorRegion:
    """Detach the mutable buffers nested in one immutable-shaped region tree."""
    raster = np.array(region.raster, copy=True)
    coverage = None if region.coverage is None else np.array(region.coverage, copy=True)
    diagnostics = _freeze(region.diagnostics)
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
            id=region.id,
            current=_copy_shape(region.current),
            original=_copy_shape(region.original),
            fill=_copy_fill(region.fill),
            z=region.z,
            footprint=region.footprint,
            raster=raster,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=region.drawing_id,
            source_regions=region.source_regions,
            coverage=coverage,
            diagnostics=diagnostics,
        )
    return VectorRegion.branch(
        id=region.id,
        children=tuple(_snapshot_region(child) for child in region.children),
        z=region.z,
        raster=raster,
        fill=_copy_fill(region.fill),
        source_label=region.source_label,
        color_hex=region.color_hex,
        drawing_id=region.drawing_id,
        source_regions=region.source_regions,
        diagnostics=diagnostics,
    )


def _snapshot_regions(regions: tuple[VectorRegion, ...]) -> tuple[VectorRegion, ...]:
    return tuple(_snapshot_region(region) for region in regions)


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
    geometry_regions = []
    for region in trace.geometry_regions:
        mask = np.array(region.mask, copy=True)
        mask.setflags(write=False)
        contours = tuple(np.array(contour, copy=True) for contour in region.contours)
        for contour in contours:
            contour.setflags(write=False)
        geometry_regions.append(
            GeometryTraceRegion(
                id=region.id,
                mask=mask,
                contours=contours,
                trace_path=region.trace_path,
            )
        )
    return TraceResult(
        width=trace.width,
        height=trace.height,
        options=trace.options,
        regions=tuple(regions),
        region_map_svg=trace.region_map_svg,
        geometry_regions=tuple(geometry_regions),
        background=dict(trace.background),
    )


def _snapshot_source_rgb(source_rgb: np.ndarray | None, trace: TraceResult) -> np.ndarray | None:
    """Detach the preprocessed visual source used by review-panel artifacts."""
    if source_rgb is None:
        return None
    if not isinstance(source_rgb, np.ndarray) or source_rgb.shape != (trace.height, trace.width, 3):
        raise TypeError("source_rgb must be an HxWx3 array matching the trace canvas")
    snapshot = np.array(source_rgb, dtype=np.uint8, copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _validate_label(label: str | None) -> None:
    if type(label) not in (str, type(None)):
        raise TypeError("label must be exactly str or None")


def _public_version(version: DrawingVersion) -> DrawingVersion:
    """Return a detached view so callers can never mutate retained state."""
    return DrawingVersion(
        id=version.id,
        parent_id=version.parent_id,
        plan=_freeze_plan(version.plan) if version.plan is not None else None,
        regions=_snapshot_regions(version.regions) if version.regions is not None else None,
        label=version.label,
        trace=_snapshot_trace(version.trace) if version.trace is not None else None,
        geometry_guide_rgb=(
            _snapshot_source_rgb(version.geometry_guide_rgb, version.trace)
            if version.geometry_guide_rgb is not None and version.trace is not None
            else None
        ),
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
        source_rgb=_snapshot_source_rgb(drawing.source_rgb, drawing.trace),
    )


class DrawingStore:
    """Retain live drawings by opaque ID until their idle timeout elapses.

    The outbound tunnel creates a fresh MCP transport session for individual
    tool calls.  ``drawing_id`` is therefore the durable, high-entropy
    capability used by trace, artifact, and refinement calls; the ``session``
    argument remains in the public methods for API compatibility with local
    transports but is not used as a storage key.
    """

    def __init__(
        self,
        *,
        idle_ttl_seconds: float = 1800,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._idle_ttl_seconds = idle_ttl_seconds
        self._now = now
        self._drawings: dict[str, _StoredDrawing] = {}
        self._lock = RLock()

    def create(
        self,
        session: object,
        trace: TraceResult,
        *,
        regions: tuple[VectorRegion, ...],
        source_rgb: np.ndarray | None = None,
    ) -> DrawingState:
        with self._lock:
            del session  # Transport sessions are request-scoped under the tunnel.
            now = self._now()
            self._evict_expired(now)
            if not isinstance(regions, tuple) or not all(isinstance(region, VectorRegion) for region in regions):
                raise TypeError("regions must be a tuple of VectorRegion roots")
            drawing = _StoredDrawing(
                id=f"drw_{secrets.token_urlsafe(18)}",
                trace=_snapshot_trace(trace),
                source_rgb=_snapshot_source_rgb(source_rgb, trace),
                versions={
                    "v0": DrawingVersion(
                        id="v0",
                        parent_id=None,
                        plan=None,
                        regions=_snapshot_regions(regions),
                        label=None,
                        trace=_snapshot_trace(trace),
                        geometry_guide_rgb=None,
                    )
                },
                child_counts={"v0": 0},
                last_access=now,
            )
            self._drawings[drawing.id] = drawing
            return _public_drawing(drawing)

    def get(
        self,
        session: object,
        drawing_id: str,
        version_id: str,
    ) -> tuple[DrawingState, DrawingVersion]:
        with self._lock:
            del session
            now = self._now()
            self._evict_expired(now)
            drawing = self._drawing_for(drawing_id)
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
        regions: tuple[VectorRegion, ...],
        label: str | None = None,
        trace: TraceResult | None = None,
        geometry_guide_rgb: np.ndarray | None = None,
        version_id: str | None = None,
    ) -> DrawingVersion:
        with self._lock:
            del session
            now = self._now()
            self._evict_expired(now)
            drawing = self._drawing_for(drawing_id)
            if base_version not in drawing.versions:
                raise DrawingNotFound from None

            _validate_label(label)
            frozen_plan = _freeze_plan(plan)
            if not isinstance(regions, tuple) or not all(isinstance(region, VectorRegion) for region in regions):
                raise TypeError("regions must be a tuple of VectorRegion roots")
            frozen_regions = _snapshot_regions(regions)
            base = drawing.versions[base_version]
            base_trace = base.trace or drawing.trace
            frozen_trace = _snapshot_trace(trace or base_trace)
            frozen_guide_rgb = _snapshot_source_rgb(
                geometry_guide_rgb if geometry_guide_rgb is not None else base.geometry_guide_rgb,
                frozen_trace,
            )
            if version_id is None:
                child_number = drawing.child_counts[base_version]
                version_id = f"{base_version}.{child_number}"
                drawing.child_counts[base_version] = child_number + 1
            elif not _VERSION_ID.fullmatch(version_id) or version_id in drawing.versions:
                raise ValueError(f"invalid or duplicate drawing version ID: {version_id}")
            version = DrawingVersion(
                id=version_id,
                parent_id=base_version,
                plan=frozen_plan,
                regions=frozen_regions,
                label=label,
                trace=frozen_trace,
                geometry_guide_rgb=frozen_guide_rgb,
            )
            drawing.versions[version_id] = version
            drawing.child_counts[version_id] = 0
            drawing.last_access = now
            return _public_version(version)

    def _drawing_for(self, drawing_id: str) -> _StoredDrawing:
        try:
            return self._drawings[drawing_id]
        except KeyError:
            raise DrawingNotFound from None

    def _evict_expired(self, now: float) -> None:
        for drawing_id, drawing in list(self._drawings.items()):
            if now - drawing.last_access >= self._idle_ttl_seconds:
                del self._drawings[drawing_id]
