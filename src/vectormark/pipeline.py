"""Top-level orchestration: raster path/array -> structured SVG string."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .color import extract_palette, quantize
from .contour import region_contours
from .emit import mirror_use, path_svg, reflect_path_d, render_svg_doc, shape_to_path_d, shape_to_svg
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .occlusion import ScenePrimitive, reconstruct_scene
from .refine import half_ellipse_cap_fit, rounded_trapezoid_fit, symmetric_fit, symmetric_polygon_fit
from .segment import segment
from .symmetry import classify_regions, detect_axis
from .types import Axis, Region


# The shared mark-level corner radius is MEASURED from the band geometry (so it
# tracks whatever mark you feed it) and applied to every segment, instead of each
# segment picking its own radius from its own height.
#
# De-antialiasing collapses the soft AA edge that carried part of the corner's
# roundness, so the hardened mask under-reads the radius by ~the AA half-width; we
# add it back as a constant pad so the emitted radius matches the *perceived*
# source rounding. The fraction-of-height value is only a fallback when no band-
# like (straight-sided) segment is measurable.
_CORNER_RADIUS_FRACTION = 0.22
_DEANTIALIAS_PAD = 2.0


def _band_fillet_radius(contour: np.ndarray, axis_x: float) -> list[float] | None:
    """Measured corner-fillet radii (top, bottom) of a band-like region, or None
    if its right side isn't a clean straight taper. The fillet inset = how far the
    flat edge falls short of the sharp corner where the side-line meets it."""
    pts = np.asarray(contour, dtype=float)
    y_top, y_bot = pts[:, 1].min(), pts[:, 1].max()
    height = y_bot - y_top
    if height < 12:
        return None
    margin = min(height * 0.30, 16)
    sel = (pts[:, 1] > y_top + margin) & (pts[:, 1] < y_bot - margin) & (pts[:, 0] > axis_x)
    edge = pts[sel]
    if len(edge) < 6:
        return None
    A = np.column_stack([edge[:, 1], np.ones(len(edge))])
    (m, b), *_ = np.linalg.lstsq(A, edge[:, 0], rcond=None)
    if np.abs(edge[:, 0] - (m * edge[:, 1] + b)).max() > 2.5:
        return None                                  # side isn't a straight taper
    out: list[float] = []
    for edge_y in (y_top, y_bot):
        corner_x = m * edge_y + b
        near = pts[np.abs(pts[:, 1] - edge_y) < 1.5]
        if len(near):
            rr = corner_x - near[:, 0].max()
            if 0 <= rr < 0.25 * height:              # a real corner fillet, not a curved cap
                out.append(float(rr))
    return out or None


@dataclass
class Options:
    epsilon: float = 1.5          # primitive/polygon recognition tolerance (px)
    max_error: float = 1.0        # Bézier fit tolerance (px)
    max_colors: int = 16
    min_region_fraction: float = 0.001  # drop regions smaller than this × image area
    flatten: bool = False
    no_symmetry: bool = False
    corner_radius: float | None = None  # shared fillet radius; None = auto from geometry


def _mark_corner_radius(regions: list[Region], axis: Axis | None) -> float:
    """One shared fillet radius for the whole mark: the median measured band-corner
    radius plus the de-antialiasing pad. Falls back to a fraction of the median
    segment height when no band-like segment is measurable (or no axis)."""
    if axis is not None:
        measured: list[float] = []
        for region in regions:
            cs = region_contours(region.mask)
            if cs:
                m = _band_fillet_radius(cs[0], axis.x)
                if m:
                    measured.extend(m)
        if measured:
            measured.sort()
            median_r = measured[len(measured) // 2]
            return round(median_r + _DEANTIALIAS_PAD, 1)
    heights = sorted(float(r.mask.any(axis=1).sum()) for r in regions if r.area > 0)
    if not heights:
        return 0.0
    return round(_CORNER_RADIUS_FRACTION * heights[len(heights) // 2], 1)


def _fit_region(region: Region, opt: Options, axis: Axis | None, corner_radius: float) -> Shape | None:
    contours = [c for c in region_contours(region.mask) if len(c) >= 3]
    if not contours:
        return None
    if len(contours) > 1:
        # holes / counters: outer + inner contours as subpaths, even-odd fill.
        # Skip primitive/polygon recognition (those see the outer ring only and
        # would fill the hole solid).
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error).params["d"]
            for c in contours
        )
        return Shape("path", {"d": d, "fill_rule": "evenodd"})

    contour = contours[0]
    shape = recognize_primitive(contour, epsilon=opt.epsilon)
    if shape is not None:
        return _snap_to_axis(shape, axis) if axis is not None else shape

    # Straddling, non-primitive region (dome, tip): fit the half-outline and
    # mirror it → exactly symmetric. (Pairs arrive with axis=None and are
    # mirrored via <use> instead; no axis → no symmetry to lock.)
    if axis is not None:
        # band-like? a clean rounded trapezoid (straight tapering sides + flat
        # top/bottom + filleted corners) beats the free half-outline fit.
        trap = rounded_trapezoid_fit(contour, axis.x, radius=corner_radius, max_error=opt.max_error)
        if trap is not None:
            return trap
        # sharp-edged symmetric polygon (a diamond, an arrow)? keep it crisp —
        # straight edges + sharp corners — instead of letting symmetric_fit curve it.
        poly = symmetric_polygon_fit(contour, axis.x, epsilon=opt.epsilon)
        if poly is not None:
            return poly
        # flat-based dome cap? a parametric half-ellipse (two convex kappa arcs)
        # beats the free half-outline fit and is inflection-free by construction.
        cap = half_ellipse_cap_fit(contour, axis.x, corner_radius=corner_radius, max_error=opt.max_error)
        if cap is not None:
            return cap
        sym = symmetric_fit(contour, axis.x, corner_radius=corner_radius,
                            epsilon=opt.epsilon, max_error=opt.max_error)
        if sym is not None:
            return sym

    shape = recognize_polygon(contour, epsilon=opt.epsilon)
    if shape is None:
        shape = fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error)
    return shape


def _snap_to_axis(shape: Shape, axis: Axis) -> Shape:
    """Force x-centre of a straddling primitive onto the axis for exact symmetry."""
    if shape.kind in ("circle", "ellipse"):
        shape.params["cx"] = axis.x
    elif shape.kind == "rect":
        shape.params["x"] = axis.x - shape.params["w"] / 2
    return shape


def idealize(image, *, options: Options | None = None) -> str:
    opt = options or Options()
    if isinstance(image, str):
        arr = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
    h, w, _ = arr.shape

    palette = extract_palette(arr, max_colors=opt.max_colors)
    q = quantize(arr, palette)
    min_area = max(16, round(opt.min_region_fraction * h * w))
    regions = segment(q, min_area=min_area)

    if not regions:
        return render_svg_doc(w, h, [])

    silhouette = np.any([r.mask for r in regions], axis=0)
    axis = None if opt.no_symmetry else detect_axis(silhouette)
    corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(regions, axis)

    reconstructed, regions = reconstruct_scene(regions, axis, (h, w))

    straddlers: list[Region]
    pairs: list[tuple[Region, Region]]
    if axis is not None:
        straddlers, pairs = classify_regions(regions, axis)
    else:
        straddlers, pairs = list(regions), []

    body: list[str] = []
    eid = 0

    # 1) reconstructed occlusion primitives + lenses, painted in their own z-order
    for elem in sorted(reconstructed, key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"]):
        if isinstance(elem, ScenePrimitive):
            shape = Shape(elem.kind, dict(elem.params))
            if opt.flatten:
                body.append(path_svg(shape_to_path_d(shape), elem.color_hex))
            else:
                body.append(shape_to_svg(shape, elem.color_hex, f"s{eid}"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            body.append(path_svg(elem.params["d"], elem.params["color_hex"]))
        eid += 1

    # 2) everything else through the existing per-region path
    drawn = [(r, False) for r in straddlers] + [(canon, True) for canon, _ in pairs]
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, is_pair in drawn:
        shape = _fit_region(region, opt, axis if not is_pair else None, corner_radius)
        if shape is None:
            continue
        if opt.flatten:
            d = shape_to_path_d(shape)
            rule = shape.params.get("fill_rule")
            body.append(path_svg(d, region.color_hex, rule))
            if is_pair and axis is not None:
                body.append(path_svg(reflect_path_d(d, axis.x), region.color_hex, rule))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(shape, region.color_hex, elem_id))
            if is_pair and axis is not None:
                body.append(mirror_use(elem_id, axis))
        eid += 1

    return render_svg_doc(w, h, body)
