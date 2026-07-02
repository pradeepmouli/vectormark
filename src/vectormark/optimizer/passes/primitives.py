from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ...fit import recognize_primitive
from ..framework import Proposal
from ..vector_region import VectorRegion


def _polygon_exterior_points(flat: object) -> np.ndarray | None:
    polygon: Polygon | None = None

    if isinstance(flat, Polygon):
        polygon = flat
    elif isinstance(flat, MultiPolygon):
        polygons = [poly for poly in flat.geoms if not poly.is_empty]
        if polygons:
            polygon = max(polygons, key=lambda poly: (float(poly.area), tuple(poly.bounds)))

    if polygon is None or polygon.is_empty:
        return None
    if polygon.interiors:
        return None

    coords = np.asarray(polygon.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    return coords


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
        points = _polygon_exterior_points(obj.footprint)
        if points is None:
            continue

        primitive = recognize_primitive(points, epsilon=epsilon)
        if primitive is None or primitive == obj.current:
            continue

        proposals.append(Proposal((obj.id,), [obj.with_current(primitive)]))

    return proposals
