"""Top-level orchestration: raster path/array -> structured SVG string."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .color import extract_palette, quantize
from .contour import region_contours
from .emit import (
    linear_gradient_def,
    mirror_use,
    path_svg,
    radial_gradient_def,
    reflect_path_d,
    render_svg_doc,
    shape_to_path_d,
    shape_to_svg,
    transform_path_d,
)
from .fit import Shape, _fmt, fit_path, recognize_polygon, recognize_primitive
from .gradient import detect_gradients
from .occlusion import ScenePrimitive, reconstruct_scene
from .refine import half_ellipse_cap_fit, rounded_trapezoid_fit, symmetric_fit, symmetric_polygon_fit
from .segment import segment
from .symmetry import classify_regions, detect_axis, detect_symmetry_rotation
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
    min_region_fraction: float = 0.02  # drop regions smaller than this × largest region
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
        #
        # A holed straddler arrives with an axis (post-classification, only
        # self-symmetric regions carry one), so fit each contour's half-outline
        # and mirror it → an exactly-symmetric counter. If any contour doesn't
        # straddle cleanly, fall back to the faithful per-contour fit.
        if axis is not None:
            halves = [
                symmetric_fit(c, axis.x, corner_radius=corner_radius,
                              epsilon=opt.epsilon, max_error=opt.max_error)
                for c in contours
            ]
            if all(s is not None for s in halves):
                d = " ".join(s.params["d"] for s in halves)
                return Shape("path", {"d": d, "fill_rule": "evenodd"})
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


def _segment_image(arr: np.ndarray, opt: Options) -> tuple[int, int, list[Region]]:
    """Quantize + segment an RGB array into flat-color regions, dropping ones too
    small to be intentional.

    The size threshold is taken relative to the *largest region* (a proxy for the
    mark's scale), not the canvas, so it is resolution-independent: padding the
    image or feeding a higher-res copy does not change which regions survive. A
    small absolute floor first removes single-pixel quantization noise."""
    h, w, _ = arr.shape
    palette = extract_palette(arr, max_colors=opt.max_colors)
    q = quantize(arr, palette)
    regions = segment(q, min_area=16)
    if regions:
        cut = opt.min_region_fraction * max(r.area for r in regions)
        regions = [r for r in regions if r.area >= cut]
    return w, h, regions


Affine = tuple[float, float, float, float, float, float]


def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str]]:
    """Detect symmetry, reconstruct occlusion, fit and emit every region in
    z-order. Operates entirely in the frame of `regions` (which may be a rectified
    frame) so the caller can wrap the result in an inverse transform.

    When `bake` is given (only in flatten mode), the inverse transform is applied
    directly to each path's coordinates instead — flatten emits pure baked geometry
    with no wrapping `<g transform>`."""
    silhouette = np.any([r.mask for r in regions], axis=0)
    axis = None if opt.no_symmetry else detect_axis(silhouette)
    corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(regions, axis)

    reconstructed, regions = reconstruct_scene(regions, axis, (h, w))

    # NOTE: silhouette/axis/corner_radius above are measured on the full pre-strip mark
    # by design — one stable mark-wide fillet radius, independent of which bands the
    # gradient pass consumes, applied uniformly to the remaining flats and the footprint.
    defs: list[str] = []
    gradient_fills: list[tuple[Region, dict]] = []
    if rgb is not None and bake is None:
        gradient_fills, regions = detect_gradients(regions, rgb)

    if axis is not None:
        straddlers, pairs, loners = classify_regions(regions, axis)
    else:
        straddlers, pairs, loners = list(regions), [], []

    def emit(d: str, fill: str, rule: str | None = None) -> str:
        return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)

    body: list[str] = []
    eid = 0

    # 1) reconstructed occlusion primitives + lenses, painted in their own z-order
    for elem in sorted(reconstructed, key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"]):
        if isinstance(elem, ScenePrimitive):
            shape = Shape(elem.kind, dict(elem.params))
            if opt.flatten:
                # an annulus is two same-winding circles: it only reads as a ring
                # under even-odd fill, so carry that rule onto the baked path too.
                rule = "evenodd" if elem.kind == "annulus" else None
                body.append(emit(shape_to_path_d(shape), elem.color_hex, rule))
            else:
                body.append(shape_to_svg(shape, elem.color_hex, f"s{eid}"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            body.append(emit(elem.params["d"], elem.params["color_hex"]))
        eid += 1

    # 2) everything else through the existing per-region path. Straddlers fit
    # half-and-mirror about the axis; pairs fit once + <use> mirror; loners
    # (asymmetric, unpaired) fit as-is with no axis so they aren't force-mirrored.
    drawn = (
        [(r, axis, False) for r in straddlers]
        + [(canon, None, True) for canon, _ in pairs]
        + [(r, None, False) for r in loners]
    )
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, fit_axis, is_pair in drawn:
        shape = _fit_region(region, opt, fit_axis, corner_radius)
        if shape is None:
            continue
        if opt.flatten:
            d = shape_to_path_d(shape)
            rule = shape.params.get("fill_rule")
            body.append(emit(d, region.color_hex, rule))
            if is_pair and axis is not None:
                body.append(emit(reflect_path_d(d, axis.x), region.color_hex, rule))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(shape, region.color_hex, elem_id))
            if is_pair and axis is not None:
                body.append(mirror_use(elem_id, axis))
        eid += 1

    # gradient-filled footprints: fit the outline with the normal recognizers, emit
    # with fill="url(#gN)" and register the gradient def. axis=None (gradient marks
    # aren't force-mirrored in this cut). Emitted after all flats/occlusion prims:
    # segmentation regions are spatially disjoint so paint order is immaterial here;
    # true behind-a-flat layering of an idealized gradient footprint is out of scope
    # in this cut.
    for footprint, model in gradient_fills:
        shape = _fit_region(footprint, opt, None, corner_radius)
        if shape is None:
            continue
        gid = f"g{len(defs)}"
        gg = model["geometry"]
        if model["kind"] == "linear":
            defs.append(linear_gradient_def(gid, gg["x1"], gg["y1"], gg["x2"], gg["y2"], model["stops"]))
        else:
            defs.append(radial_gradient_def(gid, gg["cx"], gg["cy"], gg["r"], model["stops"]))
        fill = f"url(#{gid})"
        if opt.flatten:
            body.append(emit(shape_to_path_d(shape), fill, shape.params.get("fill_rule")))
        else:
            body.append(shape_to_svg(shape, fill, f"s{eid}"))
        eid += 1

    return body, defs


def _rectify_affine(rho: float, w0: int, h0: int, rw: int, rh: int) -> Affine:
    """The SVG affine (a, b, c, d, e, f) that maps a point in the rectified frame
    back to the original: translate(-rw/2,-rh/2) → rotate(-rho) → translate(w0/2,h0/2)."""
    th = np.radians(-rho)
    cos, sin = np.cos(th), np.sin(th)
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    to_centre = np.array([[1.0, 0.0, w0 / 2], [0.0, 1.0, h0 / 2], [0.0, 0.0, 1.0]])
    from_centre = np.array([[1.0, 0.0, -rw / 2], [0.0, 1.0, -rh / 2], [0.0, 0.0, 1.0]])
    m = to_centre @ rot @ from_centre
    return (m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])


def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> str | None:
    """Rotate the image so the tilted mirror axis is vertical and idealize there.
    Non-flatten output keeps the symmetry (`<use>` mirror about the vertical axis)
    and is wrapped in one inverse-rotation `<g>`; flatten output bakes that same
    rotation into the path coordinates so no transform survives. Returns None (so
    the caller falls back to upright) if the rectified frame yields no usable
    regions or its vertical symmetry no longer registers."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None
    if detect_axis(np.any([r.mask for r in regions], axis=0)) is None:
        return None
    if opt.flatten:
        body, _ = _render_body(rw, rh, regions, opt, bake=_rectify_affine(rho, w0, h0, rw, rh))
        return render_svg_doc(w0, h0, body)
    body, _ = _render_body(rw, rh, regions, opt)
    wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
            f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
    return render_svg_doc(w0, h0, [wrap, *body, "</g>"])


def idealize(image, *, options: Options | None = None) -> str:
    opt = options or Options()
    if isinstance(image, str):
        with Image.open(image) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
    h0, w0 = arr.shape[:2]

    w, h, regions = _segment_image(arr, opt)
    if not regions:
        return render_svg_doc(w, h, [])

    # Any-axis symmetry: if the mark has no *vertical* mirror but a tilted one,
    # rectify it upright, idealize there, and wrap the body back into place — so
    # the existing vertical machinery (<use> mirror, pair dedup) does the work.
    if not opt.no_symmetry:
        silhouette = np.any([r.mask for r in regions], axis=0)
        if detect_axis(silhouette) is None:
            rho = detect_symmetry_rotation(silhouette)
            if rho is not None:
                rectified = _idealize_rectified(arr, opt, rho, w0, h0)
                if rectified is not None:
                    return rectified

    body, defs = _render_body(w, h, regions, opt, rgb=arr)
    return render_svg_doc(w, h, body, defs)
