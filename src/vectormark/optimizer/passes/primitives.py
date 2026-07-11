from __future__ import annotations

import numpy as np

from ...skia_geometry import SkPath
from ...fit import recognize_primitive
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon

_MAX_GEOMETRY_RESIDUAL = 0.04


def _polygon_exterior_points(flat: object) -> tuple[np.ndarray, SkPath] | None:
    if not isinstance(flat, SkPath) or flat.is_empty:
        return None

    geoms = flat.geoms
    if len(geoms) != 1:
        return None
    poly = geoms[0]
    if poly.interiors:
        return None

    coords = np.asarray(poly.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    return coords, flat


def _geometry_residual(original: object, candidate: object) -> float:
    try:
        scale = max(float(getattr(original, "area", 0.0)), 1.0)
        residual = float(original.symmetric_difference(candidate).area / scale)
        if residual <= _MAX_GEOMETRY_RESIDUAL:
            return residual
        if not isinstance(original, SkPath) or not isinstance(candidate, SkPath):
            return residual
        minx = min(original.bounds[0], candidate.bounds[0])
        miny = min(original.bounds[1], candidate.bounds[1])
        maxx = max(original.bounds[2], candidate.bounds[2])
        maxy = max(original.bounds[3], candidate.bounds[3])
        if maxx <= minx or maxy <= miny:
            return residual
        xs = np.linspace(minx, maxx, 65, endpoint=False) + (maxx - minx) / 130
        ys = np.linspace(miny, maxy, 65, endpoint=False) + (maxy - miny) / 130
        disagree = source = 0
        for x in xs:
            for y in ys:
                in_source = original._path.contains(float(x), float(y))
                in_candidate = candidate._path.contains(float(x), float(y))
                source += int(in_source)
                disagree += int(in_source != in_candidate)
        return float(disagree / max(source, 1))
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
