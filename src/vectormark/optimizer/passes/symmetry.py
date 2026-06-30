from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ...candidate import FlatFill
from ...fit import Shape, _fmt
from ..framework import Proposal
from ..gate import gate_ok
from ..optobject import OptObject

Axis2D = namedtuple("Axis2D", "theta cx cy")

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.03
_PERIMETER_RATIO_TOL = 0.03
_PAIR_RESIDUAL_TOL = 0.02
_SELF_RESIDUAL_TOL = 0.02


def _polygonal_flat(flat: object) -> Polygon | MultiPolygon | None:
    if isinstance(flat, (Polygon, MultiPolygon)) and not flat.is_empty:
        return flat
    return None


def _flat_fill_hex(fill: object) -> str | None:
    if isinstance(fill, FlatFill):
        return fill.hex
    return None


def _normalize_angle(theta: float) -> float:
    out = math.fmod(theta, math.pi)
    if out < 0.0:
        out += math.pi
    if abs(out) < _ANGLE_EPS or abs(out - math.pi) < _ANGLE_EPS:
        return 0.0
    return float(out)


def _axis_key(axis: Axis2D) -> tuple[float, float, float]:
    theta = _normalize_angle(float(axis.theta))
    offset = float(axis.cx * math.sin(theta) - axis.cy * math.cos(theta))
    return (round(theta, 9), round(offset, 6), round(float(axis.cx), 6))


def _within_ratio(a: float, b: float, tol: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def _residual(a: Polygon | MultiPolygon, b: Polygon | MultiPolygon) -> float:
    scale = max(float(a.area), float(b.area), 1.0)
    return float(a.symmetric_difference(b).area / scale)


def _svg_reflection_matrix(axis: Axis2D) -> tuple[float, float, float, float, float, float]:
    theta = float(axis.theta)
    ux = math.cos(theta)
    uy = math.sin(theta)
    a = 2.0 * ux * ux - 1.0
    b = 2.0 * ux * uy
    c = 2.0 * ux * uy
    d = 2.0 * uy * uy - 1.0
    e = float(axis.cx) - (a * float(axis.cx) + c * float(axis.cy))
    f = float(axis.cy) - (b * float(axis.cx) + d * float(axis.cy))
    vals = (a, b, c, d, e, f)
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in vals)


def _apply_svg_matrix(
    flat: Polygon | MultiPolygon,
    matrix: tuple[float, float, float, float, float, float],
) -> Polygon | MultiPolygon:
    a, b, c, d, e, f = matrix
    return affinity.affine_transform(flat, [a, c, b, d, e, f])


def _reflect_flat(flat: Polygon | MultiPolygon, axis: Axis2D) -> Polygon | MultiPolygon:
    return _apply_svg_matrix(flat, _svg_reflection_matrix(axis))


def _orientation_axes(flat: Polygon | MultiPolygon) -> list[float]:
    angles = [0.0, math.pi / 2.0]
    rect = flat.minimum_rotated_rectangle
    if isinstance(rect, Polygon) and not rect.is_empty:
        coords = list(rect.exterior.coords)
        for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < _ANGLE_EPS and abs(dy) < _ANGLE_EPS:
                continue
            angle = _normalize_angle(math.atan2(dy, dx))
            angles.extend([angle, _normalize_angle(angle + math.pi / 2.0)])

    out: list[float] = []
    for angle in angles:
        if all(abs(_normalize_angle(angle - existing)) > 1e-6 for existing in out):
            out.append(angle)
    return sorted(out)


def _self_axis_candidates(flat: Polygon | MultiPolygon) -> list[Axis2D]:
    centroid = flat.centroid
    axes = [Axis2D(theta, float(centroid.x), float(centroid.y)) for theta in _orientation_axes(flat)]
    return sorted(axes, key=_axis_key)


def _mirror_pair_axis(
    canonical: Polygon | MultiPolygon,
    target: Polygon | MultiPolygon,
) -> Axis2D | None:
    cc = canonical.centroid
    tc = target.centroid
    dx = float(tc.x - cc.x)
    dy = float(tc.y - cc.y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    theta = _normalize_angle(math.atan2(dy, dx) + math.pi / 2.0)
    return Axis2D(theta, float((cc.x + tc.x) / 2.0), float((cc.y + tc.y) / 2.0))


def _half_plane(flat: Polygon | MultiPolygon, axis: Axis2D) -> Polygon:
    minx, miny, maxx, maxy = flat.bounds
    span = max(maxx - minx, maxy - miny, 1.0) * 8.0
    ux = math.cos(axis.theta)
    uy = math.sin(axis.theta)
    nx = -uy
    ny = ux
    cx = float(axis.cx)
    cy = float(axis.cy)
    return Polygon(
        [
            (cx - ux * span, cy - uy * span),
            (cx + ux * span, cy + uy * span),
            (cx + ux * span + nx * span, cy + uy * span + ny * span),
            (cx - ux * span + nx * span, cy - uy * span + ny * span),
        ]
    )


def _force_self_symmetric(flat: Polygon | MultiPolygon, axis: Axis2D) -> Polygon | MultiPolygon | None:
    half = flat.intersection(_half_plane(flat, axis))
    if half.is_empty:
        return None
    reflected = _reflect_flat(half, axis)
    out = unary_union([half, reflected])
    if out.is_empty or not isinstance(out, (Polygon, MultiPolygon)):
        return None
    return out if out.is_valid else out.buffer(0)


def _ring_d(coords) -> str:
    pts = list(coords)
    if len(pts) < 4:
        return ""
    body = " ".join(f"L{_fmt(float(x))} {_fmt(float(y))}" for x, y in pts[1:])
    return f"M{_fmt(float(pts[0][0]))} {_fmt(float(pts[0][1]))} {body} Z"


def _polygon_path_parts(poly: Polygon) -> list[str]:
    parts = [_ring_d(poly.exterior.coords)]
    parts.extend(_ring_d(ring.coords) for ring in poly.interiors)
    return [part for part in parts if part]


def _geometry_to_path_shape(flat: Polygon | MultiPolygon) -> Shape | None:
    polygons = [flat] if isinstance(flat, Polygon) else list(flat.geoms)
    polygons = sorted(
        [poly for poly in polygons if not poly.is_empty],
        key=lambda poly: (round(float(poly.bounds[0]), 6), round(float(poly.bounds[1]), 6), -round(float(poly.area), 6)),
    )
    parts: list[str] = []
    has_holes = False
    for poly in polygons:
        parts.extend(_polygon_path_parts(poly))
        has_holes = has_holes or bool(poly.interiors)
    if not parts:
        return None
    params: dict[str, object] = {"d": " ".join(parts)}
    if has_holes or len(polygons) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _best_self_reconstruction(
    flat: Polygon | MultiPolygon,
    mask: np.ndarray,
) -> tuple[Axis2D, Polygon | MultiPolygon, Shape] | None:
    candidates: list[tuple[float, Axis2D, Polygon | MultiPolygon, Shape]] = []
    for axis in _self_axis_candidates(flat):
        reflected = _reflect_flat(flat, axis)
        if _residual(reflected, flat) > _SELF_RESIDUAL_TOL:
            continue
        reconstructed = _force_self_symmetric(flat, axis)
        if reconstructed is None:
            continue
        if not gate_ok(reconstructed, mask):
            continue
        shape = _geometry_to_path_shape(reconstructed)
        if shape is None:
            continue
        candidates.append((_residual(reconstructed, flat), axis, reconstructed, shape))

    if not candidates:
        return None

    _score, axis, reconstructed, shape = min(
        candidates,
        key=lambda item: (round(item[0], 12), _axis_key(item[1]), item[3].params["d"]),
    )
    return axis, reconstructed, shape


def _pair_proposal(
    canonical: OptObject,
    canonical_flat: Polygon | MultiPolygon,
    target: OptObject,
    target_flat: Polygon | MultiPolygon,
    target_fill_hex: str,
    masks: dict[int, np.ndarray],
) -> Proposal | None:
    if not _within_ratio(float(canonical_flat.area), float(target_flat.area), _AREA_RATIO_TOL):
        return None
    if not _within_ratio(float(canonical_flat.length), float(target_flat.length), _PERIMETER_RATIO_TOL):
        return None
    axis = _mirror_pair_axis(canonical_flat, target_flat)
    if axis is None:
        return None
    matrix = _svg_reflection_matrix(axis)
    reflected = _apply_svg_matrix(canonical_flat, matrix)
    if _residual(reflected, target_flat) > _PAIR_RESIDUAL_TOL:
        return None
    if not gate_ok(reflected, masks[target.id]):
        return None
    return Proposal(
        (target.id,),
        [
            OptObject(
                id=target.id,
                exact=Shape(
                    "use",
                    {
                        "href_obj_id": canonical.id,
                        "transform": matrix,
                        "fill": target_fill_hex,
                    },
                ),
                fill=target.fill,
                z=target.z,
                flat=reflected,
            )
        ],
    )


def symmetry_pass(
    objects: list[OptObject],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    usable: list[tuple[OptObject, Polygon | MultiPolygon, str | None]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        flat = _polygonal_flat(obj.flat)
        if flat is None or obj.exact.kind == "use" or obj.id not in masks:
            continue
        usable.append((obj, flat, _flat_fill_hex(obj.fill)))

    proposals: list[Proposal] = []
    paired_ids: set[int] = set()

    for index, (canonical, canonical_flat, canonical_fill) in enumerate(usable):
        if canonical.id in paired_ids:
            continue
        if canonical_fill is None:
            continue
        for target, target_flat, target_fill in usable[index + 1:]:
            if target.id in paired_ids:
                continue
            if target_fill is None:
                continue
            proposal = _pair_proposal(
                canonical,
                canonical_flat,
                target,
                target_flat,
                target_fill,
                masks,
            )
            if proposal is None:
                continue
            proposals.append(proposal)
            paired_ids.update({canonical.id, target.id})
            break

    for obj, flat, _fill_hex in usable:
        if obj.id in paired_ids:
            continue
        best = _best_self_reconstruction(flat, masks[obj.id])
        if best is None:
            continue
        _axis, reconstructed, shape = best
        if shape == obj.exact:
            continue
        proposals.append(Proposal((obj.id,), [OptObject(obj.id, shape, obj.fill, obj.z, reconstructed)]))

    return proposals
