from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from ...skia_geometry import SkPath, unary_union, affinity
from ...candidate import FlatFill
from ...contour import region_corner_radius
from ...fit import Shape, _fmt, fit_path
from ...refine import fit_path_half_fit, half_ellipse_cap_half_fit, rounded_trapezoid_half_fit, symmetric_half_fit
from ..framework import Proposal
from ..gate import rasterize
from ..shape_transform import bake_shape_transform
from ..vector_region import VectorRegion

Axis2D = namedtuple("Axis2D", "theta cx cy")

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.03
_PERIMETER_RATIO_TOL = 0.03
_PAIR_RESIDUAL_TOL = 0.02
_SELF_RESIDUAL_TOL = 0.02
_SELF_AXIS_OVERLAP = 0.75
_SELF_MIRROR_Z_OFFSET = 0.1
_MIN_SKIA_CONTOUR_AREA = 1.0
_MIN_SKIA_CONTOUR_AREA_FRACTION = 1e-4


def _polygonal_flat(flat: object) -> SkPath | None:
    if isinstance(flat, SkPath) and not flat.is_empty:
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


def _residual(a: SkPath, b: SkPath) -> float:
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
    flat: SkPath,
    matrix: tuple[float, float, float, float, float, float],
) -> SkPath:
    a, b, c, d, e, f = matrix
    return affinity.affine_transform(flat, [a, c, b, d, e, f])


def _reflect_flat(flat: SkPath, axis: Axis2D) -> SkPath:
    return _apply_svg_matrix(flat, _svg_reflection_matrix(axis))


def _orientation_axes(flat: SkPath) -> list[float]:
    angles = [0.0, math.pi / 2.0]
    rect = flat.minimum_rotated_rectangle
    if not rect.is_empty:
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


def _self_axis_candidates(flat: SkPath) -> list[Axis2D]:
    centroid = flat.centroid
    axes = [Axis2D(theta, float(centroid.x), float(centroid.y)) for theta in _orientation_axes(flat)]
    return sorted(axes, key=_axis_key)


def _mirror_pair_axis(
    canonical: SkPath,
    target: SkPath,
) -> Axis2D | None:
    cc = canonical.centroid
    tc = target.centroid
    dx = float(tc.x - cc.x)
    dy = float(tc.y - cc.y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    theta = _normalize_angle(math.atan2(dy, dx) + math.pi / 2.0)
    return Axis2D(theta, float((cc.x + tc.x) / 2.0), float((cc.y + tc.y) / 2.0))


def _half_plane(flat: SkPath, axis: Axis2D) -> SkPath:
    minx, miny, maxx, maxy = flat.bounds
    span = max(maxx - minx, maxy - miny, 1.0) * 8.0
    ux = math.cos(axis.theta)
    uy = math.sin(axis.theta)
    nx = -uy
    ny = ux
    cx = float(axis.cx)
    cy = float(axis.cy)
    seam_x = nx * _SELF_AXIS_OVERLAP
    seam_y = ny * _SELF_AXIS_OVERLAP
    return SkPath(
        shell=[
            (cx - ux * span - seam_x, cy - uy * span - seam_y),
            (cx + ux * span - seam_x, cy + uy * span - seam_y),
            (cx + ux * span + nx * span, cy + uy * span + ny * span),
            (cx - ux * span + nx * span, cy - uy * span + ny * span),
        ]
    )


def _canonical_half(flat: SkPath, axis: Axis2D) -> SkPath | None:
    half = flat.intersection(_half_plane(flat, axis))
    if half.is_empty or not isinstance(half, SkPath):
        return None
    return half.buffer(0)


def _ring_d(coords) -> str:
    pts = list(coords)
    if len(pts) < 4:
        return ""
    body = " ".join(f"L{_fmt(float(x))} {_fmt(float(y))}" for x, y in pts[1:])
    return f"M{_fmt(float(pts[0][0]))} {_fmt(float(pts[0][1]))} {body} Z"


def _coords_area(coords) -> float:
    pts = list(coords)
    if len(pts) < 4:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(area) / 2.0


def _fit_ring_path(coords, *, epsilon: float, max_error: float, cubic: bool) -> str:
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 4:
        return _ring_d(coords)
    try:
        return str(
            fit_path(
                pts,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                prefer_simple_curves=True,
            ).params["d"]
        )
    except Exception:
        return _ring_d(coords)


def _geometry_to_path_shape(
    flat: SkPath,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape | None:
    polygons = flat.geoms
    polygons = sorted(
        [poly for poly in polygons if not poly.is_empty],
        key=lambda poly: (round(float(poly.bounds[0]), 6), round(float(poly.bounds[1]), 6), -round(float(poly.area), 6)),
    )
    max_area = max((float(poly.area) for poly in polygons), default=0.0)
    contour_area_floor = max(_MIN_SKIA_CONTOUR_AREA, max_area * _MIN_SKIA_CONTOUR_AREA_FRACTION)
    parts: list[str] = []
    has_holes = False
    for poly in polygons:
        if float(poly.area) < contour_area_floor:
            continue
        parts.append(_fit_ring_path(poly.exterior.coords, epsilon=epsilon, max_error=max_error, cubic=cubic))
        kept_holes = [
            ring
            for ring in poly.interiors
            if _coords_area(ring.coords) >= contour_area_floor
        ]
        parts.extend(_fit_ring_path(ring.coords, epsilon=epsilon, max_error=max_error, cubic=cubic) for ring in kept_holes)
        has_holes = has_holes or bool(kept_holes)
    parts = [part for part in parts if part]
    if not parts:
        return None
    params: dict[str, object] = {"d": " ".join(parts)}
    if has_holes or len(polygons) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _self_symmetry_branch(
    obj: VectorRegion,
    *,
    axis: Axis2D,
    half: SkPath,
    reflected_half: SkPath,
    half_shape: Shape,
    residual: float,
    mirror_id: int,
    mask_shape: tuple[int, int],
) -> VectorRegion:
    matrix = _svg_reflection_matrix(axis)
    source = obj.with_current(
        half_shape,
        footprint=half,
        raster=rasterize(half, mask_shape),
        diagnostics=_self_symmetry_diagnostics(axis, residual),
    )
    mirror = VectorRegion(
        mirror_id,
        Shape(
            "use",
            {
                "href_obj_id": int(source.id),
                "transform": matrix,
            },
        ),
        obj.fill,
        obj.z + _SELF_MIRROR_Z_OFFSET,
        footprint=reflected_half,
        raster=rasterize(reflected_half, mask_shape),
        source_label=obj.source_label,
        color_hex=obj.color_hex,
        diagnostics={
            "symmetry": {
                "accepted": True,
                "mode": "self_mirror",
                "matched_source": int(source.id),
                "axis": {
                    "theta": float(axis.theta),
                    "cx": float(axis.cx),
                    "cy": float(axis.cy),
                },
                "residual": float(residual),
            }
        },
    )
    reconstructed = unary_union([half, reflected_half])
    return VectorRegion.branch(
        id=obj.id,
        children=[source, mirror],
        z=obj.z,
        raster=obj.raster,
        footprint=reconstructed,
        fill=obj.fill,
        source_label=obj.source_label,
        color_hex=obj.color_hex,
        diagnostics=_self_symmetry_diagnostics(axis, residual),
    )


def _self_symmetry_diagnostics(axis: Axis2D, residual: float) -> dict[str, object]:
    return {
        "symmetry": {
            "accepted": True,
            "mode": "self",
            "axis": {
                "theta": float(axis.theta),
                "cx": float(axis.cx),
                "cy": float(axis.cy),
            },
            "residual": float(residual),
        }
    }


def _fit_polygon_path(poly: SkPath, *, epsilon: float, max_error: float, cubic: bool) -> str | None:
    coords = np.asarray(poly.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    try:
        fitted = fit_path(
            coords,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            prefer_simple_curves=True,
        )
    except Exception:
        return None
    return str(fitted.params["d"])


def _fit_geometry_to_path_shape(
    flat: SkPath,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape | None:
    if not flat.is_empty and len(flat.geoms) > 1:
        return _geometry_to_path_shape(flat, epsilon=epsilon, max_error=max_error, cubic=cubic)
    if flat.is_empty:
        return None
    exterior = _fit_polygon_path(flat, epsilon=epsilon, max_error=max_error, cubic=cubic)
    if exterior is None:
        return _geometry_to_path_shape(flat, epsilon=epsilon, max_error=max_error, cubic=cubic)
    parts = [exterior]
    contour_area_floor = max(_MIN_SKIA_CONTOUR_AREA, float(flat.area) * _MIN_SKIA_CONTOUR_AREA_FRACTION)
    kept_holes = [
        ring
        for ring in flat.interiors
        if _coords_area(ring.coords) >= contour_area_floor
    ]
    parts.extend(_fit_ring_path(ring.coords, epsilon=epsilon, max_error=max_error, cubic=cubic) for ring in kept_holes)
    params: dict[str, object] = {"d": " ".join(part for part in parts if part)}
    if kept_holes:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _vertical_half_side(half: SkPath, axis: Axis2D) -> str:
    return "left" if float(half.centroid.x) < float(axis.cx) else "right"


def _symmetric_refine_half_shape(
    flat: SkPath,
    half: SkPath,
    axis: Axis2D,
    *,
    epsilon: float,
    max_error: float,
    corner_radius: float,
    cubic: bool,
) -> Shape | None:
    if flat.interiors or len(flat.geoms) != 1:
        return None
    # A half-ellipse cap is bilaterally symmetric about a vertical line.
    if abs(math.cos(float(axis.theta))) > 1e-6:
        return None
    side = _vertical_half_side(half, axis)
    contour = np.asarray(flat.exterior.coords, dtype=float)
    for build in (
        lambda: half_ellipse_cap_half_fit(contour, float(axis.cx), side=side, max_error=max_error),
        lambda: rounded_trapezoid_half_fit(
            contour,
            float(axis.cx),
            side=side,
            radius=corner_radius,
            max_error=max_error,
        ),
        lambda: symmetric_half_fit(
            contour,
            float(axis.cx),
            side=side,
            corner_radius=corner_radius,
            epsilon=epsilon,
            max_error=max_error,
        ),
        lambda: fit_path_half_fit(
            contour,
            float(axis.cx),
            side=side,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        ),
    ):
        try:
            shape = build()
        except Exception:
            continue
        if shape is not None:
            return shape
    return None


def _best_self_reconstruction(
    flat: SkPath,
    mask: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> tuple[Axis2D, SkPath, SkPath, Shape, float] | None:
    corner_radius = region_corner_radius(mask)
    candidates: list[tuple[float, Axis2D, SkPath, SkPath, Shape]] = []
    for axis in _self_axis_candidates(flat):
        reflected = _reflect_flat(flat, axis)
        if _residual(reflected, flat) > _SELF_RESIDUAL_TOL:
            continue
        half = _canonical_half(flat, axis)
        if half is None:
            continue
        reflected_half = _reflect_flat(half, axis)
        reconstructed = unary_union([half, reflected_half])
        if reconstructed is None:
            continue
        if reconstructed.is_empty or not isinstance(reconstructed, SkPath):
            continue
        shape = _symmetric_refine_half_shape(
            flat,
            half,
            axis,
            epsilon=epsilon,
            max_error=max_error,
            corner_radius=corner_radius,
            cubic=cubic,
        )
        if shape is None:
            shape = _fit_geometry_to_path_shape(half, epsilon=epsilon, max_error=max_error, cubic=cubic)
        if shape is None:
            continue
        candidates.append((_residual(reconstructed, flat), axis, half, reflected_half, shape))

    if not candidates:
        return None

    score, axis, half, reflected_half, shape = min(
        candidates,
        key=lambda item: (round(item[0], 12), _axis_key(item[1]), item[4].params["d"]),
    )
    return axis, half, reflected_half, shape, score


def _pair_proposal(
    canonical: VectorRegion,
    canonical_flat: SkPath,
    target: VectorRegion,
    target_flat: SkPath,
    masks: dict[int, np.ndarray],
) -> Proposal | None:
    del masks
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
    return Proposal(
        (canonical.id, target.id),
        [
            canonical,
            target.with_current(
                bake_shape_transform(canonical.current, matrix),
                footprint=reflected,
                diagnostics={
                    "symmetry": {
                        "accepted": True,
                        "mode": "pair",
                        "matched_source": int(canonical.id),
                        "axis": {
                            "theta": float(axis.theta),
                            "cx": float(axis.cx),
                            "cy": float(axis.cy),
                        },
                    }
                },
            )
        ],
    )


def symmetry_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> list[Proposal]:
    usable: list[tuple[VectorRegion, SkPath]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        flat = _polygonal_flat(obj.footprint)
        if flat is None or obj.current.kind == "use" or obj.id not in masks:
            continue
        usable.append((obj, flat))

    proposals: list[Proposal] = []
    paired_ids: set[int] = set()
    next_id = max((int(obj.id) for obj, _flat in usable), default=-1) + 1

    for index, (canonical, canonical_flat) in enumerate(usable):
        if canonical.id in paired_ids:
            continue
        for target, target_flat in usable[index + 1:]:
            if target.id in paired_ids:
                continue
            proposal = _pair_proposal(
                canonical,
                canonical_flat,
                target,
                target_flat,
                masks,
            )
            if proposal is None:
                continue
            proposals.append(proposal)
            paired_ids.update({canonical.id, target.id})
            break

    for obj, flat in usable:
        if obj.id in paired_ids:
            continue
        if obj.current.kind != "path":
            continue
        best = _best_self_reconstruction(flat, masks[obj.id], epsilon=epsilon, max_error=max_error, cubic=cubic)
        if best is None:
            continue
        axis, half, reflected_half, shape, residual = best
        proposals.append(
            Proposal(
                (obj.id,),
                [
                    _self_symmetry_branch(
                        obj,
                        axis=axis,
                        half=half,
                        reflected_half=reflected_half,
                        half_shape=shape,
                        residual=residual,
                        mirror_id=next_id,
                        mask_shape=masks[obj.id].shape,
                    )
                ],
            )
        )
        next_id += 1

    return proposals
