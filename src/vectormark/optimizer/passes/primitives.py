from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ...fit import recognize_primitive
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon

_MAX_GEOMETRY_RESIDUAL = 0.04


def _polygon_exterior_points(flat: object) -> tuple[np.ndarray, Polygon] | None:
    polygon: Polygon | None = None

    if isinstance(flat, Polygon):
        polygon = flat
    elif isinstance(flat, MultiPolygon):
        return None

    if polygon is None or polygon.is_empty:
        return None
    if polygon.interiors:
        return None

    coords = np.asarray(polygon.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    return coords, polygon


def _geometry_residual(original: object, candidate: object) -> float:
    try:
        scale = max(float(getattr(original, "area", 0.0)), 1.0)
        return float(original.symmetric_difference(candidate).area / scale)
    except Exception:
        return float("inf")


def primitives_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
) -> list[Proposal]:
    del masks

    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        polygonal = _polygon_exterior_points(obj.footprint)
        if polygonal is None:
            continue
        points, fit_geometry = polygonal

        primitive = recognize_primitive(points, epsilon=epsilon)
        if primitive is None or primitive == obj.current:
            continue
        if _geometry_residual(fit_geometry, to_polygon(primitive)) > _MAX_GEOMETRY_RESIDUAL:
            continue

        proposals.append(Proposal((obj.id,), [obj.with_current(primitive)]))

    return proposals
