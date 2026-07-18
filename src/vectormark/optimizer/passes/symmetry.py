from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from ...skia_geometry import SkPath, unary_union, affinity
from ...candidate import FlatFill
from ...contour import region_corner_radius
from ...fit import Shape, _fmt, fit_path
from ...refine import fit_path_half_fit, half_ellipse_cap_half_fit, symmetric_half_fit
from ..framework import Proposal
from ..gate import rasterize
from ..shape_transform import bake_shape_transform
from ..vector_region import VectorRegion, _parse_subpaths, to_polygon

Axis2D = namedtuple("Axis2D", "theta cx cy")

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.03
_PERIMETER_RATIO_TOL = 0.03
_PAIR_RESIDUAL_TOL = 0.02
# Candidate threshold only.  A proposed symmetric reconstruction must still
# pass the fitted-geometry and framework scene-fidelity checks below.
_SELF_RESIDUAL_TOL = 0.04
_FITTED_RECONSTRUCTION_RESIDUAL_TOL = 0.04
_SELF_AXIS_OVERLAP = 0.75
_SELF_MIRROR_Z_OFFSET = 0.1
_MIN_SKIA_CONTOUR_AREA = 1.0
_MIN_SKIA_CONTOUR_AREA_FRACTION = 1e-4
_MAX_AUTO_SYMMETRY_INTERIORS = 8
_MAX_AUTO_SYMMETRY_COMPONENTS = 4


def _polygonal_flat(flat: object) -> SkPath | None:
    if isinstance(flat, SkPath) and not flat.is_empty:
        return flat
    return None


def _automatic_symmetry_is_tractable(flat: SkPath) -> bool:
    """Keep automatic reconstruction bounded on noisy compound trace roots.

    A region with many counters/components has ambiguous semantic symmetry and
    makes each reflected-union residual disproportionately expensive.  The
    original path remains intact; callers may still apply explicit symmetry
    relationships when that intent is known.
    """
    return len(flat.interiors) <= _MAX_AUTO_SYMMETRY_INTERIORS and len(flat.geoms) <= _MAX_AUTO_SYMMETRY_COMPONENTS


def _is_material_surface(obj: VectorRegion) -> bool:
    """Whether this leaf is a fill partition rather than a semantic shape."""
    return isinstance(obj.diagnostics.get("material_surface"), dict)


def _is_material_surface_root(obj: VectorRegion) -> bool:
    """Whether a branch owns one semantic outline partitioned into materials."""
    return (
        obj.is_branch
        and isinstance(obj.diagnostics.get("geometry_seed"), dict)
        and len(obj.children) > 1
        and all(child.is_leaf and _is_material_surface(child) for child in obj.children)
    )


def _is_geometry_root_leaf(obj: VectorRegion) -> bool:
    """Whether a single material leaf is itself the trace-owned geometry root."""
    return (
        obj.is_leaf
        and obj.drawing_id is not None
        and isinstance(obj.diagnostics.get("geometry_seed"), dict)
    )


def _geometry_root_footprint(obj: VectorRegion, fallback: SkPath) -> SkPath:
    """Prefer a root's immutable trace seed over intermediate leaf rewrites."""
    if not _is_geometry_root_leaf(obj) or obj.original is None:
        return fallback
    seeded = to_polygon(obj.original)
    return seeded if not seeded.is_empty else fallback


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


def _project_to_axis(point: tuple[float, float], axis: Axis2D) -> tuple[float, float]:
    """Orthogonally project a point onto a symmetry axis."""
    ux, uy = math.cos(float(axis.theta)), math.sin(float(axis.theta))
    dx, dy = point[0] - float(axis.cx), point[1] - float(axis.cy)
    distance = dx * ux + dy * uy
    return (float(axis.cx) + distance * ux, float(axis.cy) + distance * uy)


def _endpoint_indices(command: str) -> tuple[int, int] | None:
    return {
        "M": (0, 1),
        "L": (0, 1),
        "Q": (2, 3),
        "C": (4, 5),
        "A": (5, 6),
    }.get(command)


def _pin_half_shape_to_axis(shape: Shape, axis: Axis2D, *, tolerance: float = 2.0) -> Shape | None:
    """Make a fitted half close exactly along its symmetry axis.

    The fitters intentionally focus on the exterior boundary.  Their closure
    points can still be sub-pixel-adjacent to the cut, which becomes a visible
    gap after mirroring.  Project only the start and final endpoint; SVG's Z
    then supplies a perfectly straight, coincident axis edge for Skia to union.
    """
    if shape.kind != "path":
        return None
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    if len(subpaths) != 1 or len(subpaths[0]) < 3 or subpaths[0][0][0] != "M":
        return None

    commands = [(command, list(values)) for command, values in subpaths[0]]
    final_index = next((index for index in range(len(commands) - 1, -1, -1) if _endpoint_indices(commands[index][0]) is not None), None)
    if final_index is None or final_index == 0:
        return None

    def pin(index: int) -> bool:
        command, values = commands[index]
        endpoint = _endpoint_indices(command)
        assert endpoint is not None
        current = (float(values[endpoint[0]]), float(values[endpoint[1]]))
        projected = _project_to_axis(current, axis)
        if math.dist(current, projected) > tolerance:
            return False
        values[endpoint[0]], values[endpoint[1]] = projected
        return True

    if not pin(0) or not pin(final_index):
        return None

    def command_d(command: str, values: list[float]) -> str:
        return command if not values else f"{command}{' '.join(_fmt(value) for value in values)}"

    params = dict(shape.params)
    params["d"] = " ".join(command_d(command, values) for command, values in commands)
    return Shape("path", params)


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
    half_shape = _pin_half_shape_to_axis(half_shape, axis) or half_shape
    fitted_half = to_polygon(half_shape)
    if fitted_half.is_empty:
        fitted_half = half
    fitted_reflected_half = _apply_svg_matrix(fitted_half, matrix)
    if fitted_reflected_half.is_empty:
        fitted_reflected_half = reflected_half
    source = obj.with_current(
        half_shape,
        footprint=fitted_half,
        raster=rasterize(fitted_half, mask_shape),
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
        footprint=fitted_reflected_half,
        raster=rasterize(fitted_reflected_half, mask_shape),
        source_label=obj.source_label,
        color_hex=obj.color_hex,
        drawing_id=obj.drawing_id,
        source_regions=obj.source_regions,
        coverage=obj.coverage,
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
    reconstructed = unary_union([fitted_half, fitted_reflected_half])
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


def _intrinsic_symmetry_diagnostics(axis: Axis2D, residual: float) -> dict[str, object]:
    """Report verified symmetry without replacing an explicit full geometry."""
    return {
        "symmetry": {
            "accepted": True,
            "mode": "intrinsic",
            "axis": {
                "theta": float(axis.theta),
                "cx": float(axis.cx),
                "cy": float(axis.cy),
            },
            "residual": float(residual),
        }
    }


def _best_self_axis(flat: SkPath) -> tuple[Axis2D, float] | None:
    """Find a valid reflection axis without changing the represented geometry."""
    candidates = [
        (_residual(_reflect_flat(flat, axis), flat), axis)
        for axis in _self_axis_candidates(flat)
    ]
    candidates = [candidate for candidate in candidates if candidate[0] <= _SELF_RESIDUAL_TOL]
    if not candidates:
        return None
    residual, axis = min(candidates, key=lambda item: (round(item[0], 12), _axis_key(item[1])))
    return axis, residual


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


def _symmetric_refine_half_shapes(
    flat: SkPath,
    half: SkPath,
    axis: Axis2D,
    *,
    epsilon: float,
    max_error: float,
    corner_radius: float,
    cubic: bool,
) -> tuple[Shape, ...]:
    if flat.interiors or len(flat.geoms) != 1:
        return ()
    # A half-ellipse cap is bilaterally symmetric about a vertical line.
    if abs(math.cos(float(axis.theta))) > 1e-6:
        return ()
    side = _vertical_half_side(half, axis)
    contour = np.asarray(flat.exterior.coords, dtype=float)
    shapes: list[Shape] = []
    for build in (
        lambda: half_ellipse_cap_half_fit(contour, float(axis.cx), side=side, max_error=max_error),
        lambda: symmetric_half_fit(
            contour,
            float(axis.cx),
            side=side,
            corner_radius=corner_radius,
            epsilon=epsilon,
            max_error=max_error,
        ),
    ):
        try:
            shape = build()
        except Exception:
            continue
        if shape is not None:
            shapes.append(shape)
    return tuple(shapes)


def _best_self_reconstruction(
    flat: SkPath,
    mask: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    preserve_explicit_geometry: bool = False,
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
        shapes: tuple[Shape, ...] = ()
        if not preserve_explicit_geometry:
            shapes = _symmetric_refine_half_shapes(
                flat,
                half,
                axis,
                epsilon=epsilon,
                max_error=max_error,
                corner_radius=corner_radius,
                cubic=cubic,
            )
        def accept(shape: Shape) -> tuple[float, Shape] | None:
            pinned = _pin_half_shape_to_axis(shape, axis)
            if pinned is None:
                return None
            fitted_half = to_polygon(pinned)
            if fitted_half.is_empty:
                return None
            fitted_reconstructed = unary_union([fitted_half, _reflect_flat(fitted_half, axis)])
            fitted_residual = _residual(fitted_reconstructed, flat)
            if fitted_residual > _FITTED_RECONSTRUCTION_RESIDUAL_TOL:
                return None
            return fitted_residual, pinned

        accepted = next((result for shape in shapes if (result := accept(shape)) is not None), None)
        if accepted is None:
            side = _vertical_half_side(half, axis)
            contour = np.asarray(flat.exterior.coords, dtype=float)
            generic_shapes = [
                fit_path_half_fit(
                    contour,
                    float(axis.cx),
                    side=side,
                    epsilon=epsilon,
                    max_error=max_error,
                    cubic=cubic,
                ),
                _fit_geometry_to_path_shape(half, epsilon=epsilon, max_error=max_error, cubic=cubic),
            ]
            accepted = next(
                (result for shape in generic_shapes if shape is not None and (result := accept(shape)) is not None),
                None,
            )
        if accepted is not None:
            fitted_residual, pinned = accepted
            candidates.append((fitted_residual, axis, half, reflected_half, pinned))

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


def _material_root_self_symmetry_proposal(
    obj: VectorRegion,
    flat: SkPath,
    mask: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Proposal | None:
    """Mirror a semantic root, then reclip its material children to that outline.

    A drawing trace keeps continuous fills as child regions so later fill work
    can remain localized.  Those children are not the object whose exterior is
    symmetric; applying the leaf-only reconstruction to them either changes a
    colour boundary or does nothing.  Reconstruct the seed footprint once and
    preserve every child as an intersection with the new exterior instead.
    """
    best = _best_self_reconstruction(
        flat,
        mask,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
    )
    if best is None:
        return None
    axis, _half, _reflected_half, half_shape, residual = best
    fitted_half = to_polygon(half_shape)
    if fitted_half.is_empty:
        return None
    reconstructed = unary_union([fitted_half, _reflect_flat(fitted_half, axis)])
    if not isinstance(reconstructed, SkPath) or reconstructed.is_empty:
        return None

    children: list[VectorRegion] = []
    for child in obj.children:
        child_flat = _polygonal_flat(child.footprint)
        if child_flat is None:
            return None
        clipped = child_flat.intersection(reconstructed)
        if clipped.is_empty:
            return None
        shape = _fit_geometry_to_path_shape(clipped, epsilon=epsilon, max_error=max_error, cubic=cubic)
        if shape is None:
            return None
        children.append(
            child.with_current(
                shape,
                footprint=clipped,
                raster=rasterize(clipped, mask.shape),
                diagnostics={
                    "symmetry": {
                        "accepted": True,
                        "mode": "root_clip",
                        "root": int(obj.id),
                    }
                },
            )
        )

    return Proposal(
        (obj.id,),
        [
            VectorRegion.branch(
                id=obj.id,
                children=children,
                z=obj.z,
                raster=rasterize(reconstructed, mask.shape),
                footprint=reconstructed,
                fill=obj.fill,
                source_label=obj.source_label,
                color_hex=obj.color_hex,
                drawing_id=obj.drawing_id,
                source_regions=obj.source_regions,
                diagnostics={
                    "symmetry": {
                        "accepted": True,
                        "mode": "root_self",
                        "axis": {
                            "theta": float(axis.theta),
                            "cx": float(axis.cx),
                            "cy": float(axis.cy),
                        },
                        "residual": float(residual),
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
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not _is_material_surface_root(obj) or obj.id not in masks:
            continue
        # A material-surface root renders its children, not its own footprint.
        # Mirroring the root then clipping/re-fitting the original children can
        # emit a scene that is less symmetric than the input despite the root
        # footprint itself being symmetric.  Until geometry and material layers
        # have distinct render representations, automatic self-symmetry must
        # fail closed for these roots.  Explicit plans may still reconstruct
        # their geometry with a deliberate fill assignment.
        continue

    usable: list[tuple[VectorRegion, SkPath]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        # Compound splitting creates implementation children for nested
        # cutouts.  They are not independently trace-owned semantic shapes,
        # so automatic self-reflection must never reconstruct them directly.
        if isinstance(obj.diagnostics.get("compound"), dict):
            continue
        flat = _polygonal_flat(obj.footprint)
        if flat is None or obj.current.kind == "use" or obj.id not in masks:
            continue
        if not _automatic_symmetry_is_tractable(flat):
            continue
        usable.append((obj, flat))

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
        # ``material_surface`` leaves partition one geometry root by colour.
        # They are not independent semantic shapes, so a self-reflection can
        # preserve their footprint while materially changing the rendered
        # artwork.  Automatic symmetry belongs on geometry roots; callers may
        # still apply an explicit symmetry plan to a selected material leaf.
        geometry_root_leaf = _is_geometry_root_leaf(obj)
        if _is_material_surface(obj) and not geometry_root_leaf:
            continue
        # A gradient is fitted onto finalized geometry.  It can carry a
        # verified symmetry relationship, but must not cause a self-symmetry
        # rewrite that replaces that geometry with a mirrored half.
        if not isinstance(obj.fill, FlatFill) and not geometry_root_leaf:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(
                    Proposal(
                        (obj.id,),
                        [obj.with_current(obj.current, diagnostics=_intrinsic_symmetry_diagnostics(axis, residual))],
                    )
                )
            continue
        # Native primitives already encode exact geometric symmetry.  Verify
        # and report the relation without replacing their deterministic form.
        if obj.current.kind in {"circle", "ellipse", "rect"}:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(
                    Proposal(
                        (obj.id,),
                        [obj.with_current(obj.current, diagnostics=_intrinsic_symmetry_diagnostics(axis, residual))],
                    )
                )
            continue
        geometry = obj.diagnostics.get("geometry") if isinstance(obj.diagnostics, dict) else None
        explicit_kind = geometry.get("explicit") if isinstance(geometry, dict) else None
        if explicit_kind in {"circle", "ellipse", "rect", "rounded_rect", "cap", "trapezoid", "rounded_trapezoid"}:
            intrinsic = _best_self_axis(flat)
            if intrinsic is not None:
                axis, residual = intrinsic
                proposals.append(
                    Proposal(
                        (obj.id,),
                        [obj.with_current(obj.current, diagnostics=_intrinsic_symmetry_diagnostics(axis, residual))],
                    )
                )
            continue
        if obj.current.kind != "path":
            continue
        preserve_explicit_geometry = explicit_kind is not None
        reconstruction_flat = _geometry_root_footprint(obj, flat)
        best = _best_self_reconstruction(
            reconstruction_flat,
            masks[obj.id],
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            preserve_explicit_geometry=preserve_explicit_geometry,
        )
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
