from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon

from ...candidate import FlatFill
from ...fit import Shape
from ..framework import Proposal
from ..gate import gate_ok
from ..vector_region import VectorRegion

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.01
_PERIMETER_RATIO_TOL = 0.01
_GEOM_RESIDUAL_TOL = 0.01


def _polygonal_flat(flat: object) -> Polygon | MultiPolygon | None:
    if isinstance(flat, (Polygon, MultiPolygon)) and not flat.is_empty:
        return flat
    return None


def _flat_fill_hex(fill: object) -> str | None:
    if isinstance(fill, FlatFill):
        return fill.hex
    return None


def _shape_descriptor(flat: Polygon | MultiPolygon) -> tuple[float, float, int]:
    return (float(flat.area), float(flat.length), len(getattr(flat, "geoms", ()) or []))


def _bucket_key(descriptor: tuple[float, float, int]) -> tuple[int, int, int]:
    area, perimeter, parts = descriptor
    return (int(round(area)), int(round(perimeter)), parts)


def _within_ratio(a: float, b: float, tol: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def _normalize_angle(theta: float) -> float:
    out = math.fmod(theta + math.pi, 2.0 * math.pi)
    if out < 0.0:
        out += 2.0 * math.pi
    out -= math.pi
    if abs(out) < _ANGLE_EPS:
        return 0.0
    return out


def _orientation_candidates(flat: Polygon | MultiPolygon) -> list[float]:
    rect = flat.minimum_rotated_rectangle
    if rect.is_empty or not isinstance(rect, Polygon):
        return [0.0]

    coords = list(rect.exterior.coords)
    angles: list[float] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < _ANGLE_EPS and abs(dy) < _ANGLE_EPS:
            continue
        angle = _normalize_angle(math.atan2(dy, dx))
        if all(abs(_normalize_angle(angle - existing)) > 1e-6 for existing in angles):
            angles.append(angle)
    return angles or [0.0]


def _matrix_for_rotation_and_centroids(
    canonical_centroid: tuple[float, float],
    target_centroid: tuple[float, float],
    theta: float,
) -> tuple[float, float, float, float, float, float]:
    a = math.cos(theta)
    b = math.sin(theta)
    c = -math.sin(theta)
    d = math.cos(theta)
    cx, cy = canonical_centroid
    tx, ty = target_centroid
    e = tx - (a * cx + c * cy)
    f = ty - (b * cx + d * cy)
    vals = (a, b, c, d, e, f)
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in vals)


def _apply_svg_matrix(
    flat: Polygon | MultiPolygon,
    matrix: tuple[float, float, float, float, float, float],
) -> Polygon | MultiPolygon:
    a, b, c, d, e, f = matrix
    return affinity.affine_transform(flat, [a, c, b, d, e, f])


def _best_transform(
    canonical: Polygon | MultiPolygon,
    target: Polygon | MultiPolygon,
) -> tuple[tuple[float, float, float, float, float, float], Polygon | MultiPolygon] | None:
    canonical_centroid = (float(canonical.centroid.x), float(canonical.centroid.y))
    target_centroid = (float(target.centroid.x), float(target.centroid.y))
    canonical_angles = _orientation_candidates(canonical)
    target_angles = _orientation_candidates(target)

    candidates: list[tuple[float, tuple[float, float, float, float, float, float], Polygon | MultiPolygon]] = []
    for canon_angle in canonical_angles:
        for target_angle in target_angles:
            theta = _normalize_angle(target_angle - canon_angle)
            matrix = _matrix_for_rotation_and_centroids(canonical_centroid, target_centroid, theta)
            transformed = _apply_svg_matrix(canonical, matrix)
            scale = max(float(target.area), float(transformed.area), 1.0)
            residual = float(transformed.symmetric_difference(target).area / scale)
            candidates.append((residual, matrix, transformed))

    if not candidates:
        return None

    residual, matrix, transformed = min(
        candidates,
        key=lambda item: (
            round(item[0], 12),
            abs(_normalize_angle(math.atan2(item[1][1], item[1][0]))),
            tuple(round(v, 12) for v in item[1]),
        ),
    )
    if residual > _GEOM_RESIDUAL_TOL:
        return None
    return matrix, transformed


def clones_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    usable: list[tuple[VectorRegion, Polygon | MultiPolygon, tuple[float, float, int], str]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        flat = _polygonal_flat(obj.footprint)
        fill_hex = _flat_fill_hex(obj.fill)
        if flat is None or fill_hex is None or obj.current.kind == "use":
            continue
        usable.append((obj, flat, _shape_descriptor(flat), fill_hex))

    by_bucket: dict[tuple[int, int, int], list[tuple[VectorRegion, Polygon | MultiPolygon, tuple[float, float, int], str]]] = defaultdict(list)
    for item in usable:
        by_bucket[_bucket_key(item[2])].append(item)

    proposals: list[Proposal] = []
    for bucket in sorted(by_bucket):
        group = by_bucket[bucket]
        queued_targets: set[int] = set()
        for index, (target_obj, target_flat, target_desc, target_fill_hex) in enumerate(group):
            matched = False
            for canonical_obj, canonical_flat, canonical_desc, _canonical_fill_hex in group[:index]:
                if canonical_obj.id in queued_targets:
                    continue
                if _canonical_fill_hex != target_fill_hex:
                    continue
                if not _within_ratio(canonical_desc[0], target_desc[0], _AREA_RATIO_TOL):
                    continue
                if not _within_ratio(canonical_desc[1], target_desc[1], _PERIMETER_RATIO_TOL):
                    continue

                best = _best_transform(canonical_flat, target_flat)
                if best is None:
                    continue
                matrix, transformed = best
                if not gate_ok(transformed, masks[target_obj.id]):
                    continue

                proposals.append(
                    Proposal(
                        (target_obj.id,),
                        [
                            VectorRegion(
                                id=target_obj.id,
                                current=Shape(
                                    "use",
                                    {
                                        "href_obj_id": canonical_obj.id,
                                        "transform": matrix,
                                        "fill": target_fill_hex,
                                    },
                                ),
                                fill=target_obj.fill,
                                z=target_obj.z,
                                footprint=transformed,
                            )
                        ],
                    )
                )
                queued_targets.add(int(target_obj.id))
                matched = True
                break
            if matched:
                continue

    return proposals
