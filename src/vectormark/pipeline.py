"""Top-level orchestration: raster path/array -> structured SVG string."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .color import extract_palette, quantize
from .contour import region_contours
from .emit import (
    apply_affine_point,
    fill_rule_for,
    mirror_use,
    path_svg,
    reflect_path_d,
    render_svg_doc,
    resolve_fill,
    shape_to_path_d,
    shape_to_svg,
    transform_path_d,
)
from .candidate import Candidate, Fill, FlatFill, LinearGradientFill, RadialGradientFill
from .components import decompose_components
from .fit import Shape, _fmt
from .gradient import detect_gradients
from .occlusion import ScenePrimitive, reconstruct_scene
from .segment import segment
from .selection import SelectionPolicy
from .selector import select_geometry
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
    fidelity_tol: float = 0.06        # selector's render-ΔE gate (slice 4a)
    selection: SelectionPolicy | None = None  # manual candidate selection (slice 4b)


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


@dataclass(frozen=True)
class AxisLine:
    """A detected mirror axis as a segment in output-frame (viewBox) coords."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class IdealizeReport:
    """What the pipeline actually emitted for one idealize() run: the histogram of
    fitter strategies the scorer chose per region, the gradient-fill count, the total
    emitted element count, and the detected mirror axes (one segment per component
    with a vertical mirror, in output-frame coords). Diagnostic annotation."""

    strategies: Mapping[str, int]
    gradients: int
    elements: int
    axes: tuple[AxisLine, ...]

    @staticmethod
    def empty() -> "IdealizeReport":
        return IdealizeReport(types.MappingProxyType({}), 0, 0, ())


def _map_axis(a: AxisLine, affine: Affine) -> AxisLine:
    x1, y1 = apply_affine_point(affine, a.x1, a.y1)
    x2, y2 = apply_affine_point(affine, a.x2, a.y2)
    return AxisLine(x1, y1, x2, y2)


def _build_report(cands: list[Candidate], axes: list[AxisLine]) -> IdealizeReport:
    strategies: dict[str, int] = {}
    gradients = 0
    for c in cands:
        if c.source == "gradient":
            gradients += 1
        if c.strategy is not None:                 # None for occlusion / lens / gradient
            strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    return IdealizeReport(types.MappingProxyType(dict(strategies)), gradients, len(cands), tuple(axes))


def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
    loners: list[Region], gradient_fills: list[tuple[Region, dict]],
    opt: Options, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray | None, *, base: int = 0,
) -> list[Candidate]:
    """Decide geometry + fill per element and return the candidate list in exact
    paint order: occlusion (by z) -> regions (by area desc) -> gradients (detect
    order). Elements whose geometry fit returns None are dropped (matching the
    old per-loop `continue`, so the emit-time id sequence is unchanged)."""
    cands: list[Candidate] = []

    for elem in sorted(
        reconstructed,
        key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"],
    ):
        if isinstance(elem, ScenePrimitive):
            cands.append(Candidate(Shape(elem.kind, dict(elem.params)),
                                   FlatFill(elem.color_hex), "occlusion"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            cands.append(Candidate(Shape("path", {"d": elem.params["d"]}),
                                   FlatFill(elem.params["color_hex"]), "lens"))

    drawn = (
        [(r, axis, False) for r in straddlers]
        + [(canon, None, True) for canon, _ in pairs]
        + [(r, None, False) for r in loners]
    )
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, fit_axis, is_pair in drawn:
        # eid = sN where N = base + current cands length. The occlusion/lens loop already
        # filled cands[0..]; a None-return below skips the append, so only emitted
        # elements consume an id — exactly matching the SVG emit-loop's GLOBAL id sequence
        # (base = candidates from prior components, so per-component lookups address sN).
        eid = f"s{base + len(cands)}"
        element = opt.selection.for_id(eid) if opt.selection is not None else None
        shape, strategy = select_geometry(region, opt, fit_axis, corner_radius, source_rgb,
                                          element=element, eid=eid)
        if shape is None:
            continue
        cands.append(Candidate(shape, FlatFill(region.color_hex), "region",
                               mirror=axis if is_pair else None, strategy=strategy))

    # Gradient footprints paint after all flats/occlusion. _expand_footprint can
    # grow a footprint over former-background pixels, so strict spatial
    # disjointness no longer holds; non-matching elements survive as even-odd
    # holes / separate flats, keeping paint order safe. True behind-a-flat
    # layering of a gradient footprint is out of scope.
    for footprint, model in gradient_fills:
        # Same sN scheme as the region loop (shared base + cands counter; gradients emit last).
        eid = f"s{base + len(cands)}"
        element = opt.selection.for_id(eid) if opt.selection is not None else None
        shape, _strategy = select_geometry(footprint, opt, None, corner_radius, source_rgb,
                                           element=element, eid=eid)
        if shape is None:
            continue
        g = model["geometry"]
        fill: Fill = (
            LinearGradientFill(g, model["stops"]) if model["kind"] == "linear"
            else RadialGradientFill(g, model["stops"])
        )
        cands.append(Candidate(shape, fill, "gradient"))

    return cands


def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str], list[Candidate], list[AxisLine]]:
    """Decompose regions into gutter-separated components, then per component detect
    symmetry, reconstruct occlusion, and fit regions — accumulating candidates that the
    single emit loop renders in z-order with globally continuous sN ids. Operates
    entirely in the frame of `regions` (which may be a rectified frame) so the caller
    can wrap the result in an inverse transform.

    When `bake` is given (only in flatten mode), the inverse transform is applied
    directly to each path's coordinates instead — flatten emits pure baked geometry
    with no wrapping `<g transform>`."""
    components = decompose_components(regions, (h, w))
    defs: list[str] = []
    cands: list[Candidate] = []
    frame_axes: list[AxisLine] = []
    for comp in components:
        silhouette = np.any([r.mask for r in comp], axis=0)
        axis = None if opt.no_symmetry else detect_axis(silhouette)
        if axis is not None:
            ys = np.nonzero(silhouette)[0]
            frame_axes.append(AxisLine(float(axis.x), float(ys.min()), float(axis.x), float(ys.max())))
        corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(comp, axis)

        reconstructed, comp = reconstruct_scene(comp, axis, (h, w))

        # Per component: one local axis + fillet radius, its own occlusion/gradient pass.
        # (Single-component marks take this loop exactly once -> identical to pre-slice-5.)
        gradient_fills: list[tuple[Region, dict]] = []
        if rgb is not None:
            gradient_fills, comp = detect_gradients(comp, rgb)

        if axis is not None:
            straddlers, pairs, loners = classify_regions(comp, axis)
        else:
            straddlers, pairs, loners = list(comp), [], []

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, corner_radius, rgb,
            base=len(cands),
        )

    def emit(d: str, fill: str, rule: str | None = None) -> str:
        return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)

    def _fill_attr(fill: Fill) -> str:
        baked = None
        if not isinstance(fill, FlatFill) and bake is not None:
            kind = "linear" if isinstance(fill, LinearGradientFill) else "radial"
            baked = _bake_gradient_geometry(fill.geometry, kind, bake)
        return resolve_fill(fill, defs, geometry=baked)

    body: list[str] = []
    eid = 0
    for cand in cands:
        geom = cand.geometry
        fill = _fill_attr(cand.fill)
        if opt.flatten:
            d = shape_to_path_d(geom)
            rule = fill_rule_for(geom)
            body.append(emit(d, fill, rule))
            if cand.mirror is not None:
                body.append(emit(reflect_path_d(d, cand.mirror.x), fill, rule))
        elif cand.source == "lens":
            body.append(emit(geom.params["d"], fill))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(geom, fill, elem_id))
            if cand.mirror is not None:
                body.append(mirror_use(elem_id, cand.mirror))
        eid += 1

    return body, defs, cands, frame_axes


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


def _bake_gradient_geometry(geom: dict, kind: str, bake: Affine) -> dict:
    """Map gradient geometry from the rectified frame to the original via the bake affine
    (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f. The rectify affine is a rigid
    rotation+translation, so the radial radius is preserved."""
    if kind == "linear":
        x1, y1 = apply_affine_point(bake, geom["x1"], geom["y1"])
        x2, y2 = apply_affine_point(bake, geom["x2"], geom["y2"])
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    cx, cy = apply_affine_point(bake, geom["cx"], geom["cy"])
    return {"cx": cx, "cy": cy, "r": geom["r"]}


def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> tuple[str | None, list[Candidate], list[AxisLine]]:
    """Rotate the image so the tilted mirror axis is vertical and idealize there.
    Non-flatten output keeps the symmetry (`<use>` mirror about the vertical axis)
    and is wrapped in one inverse-rotation `<g>`; flatten output bakes that same
    rotation into the path coordinates so no transform survives. Returns (None, [], [])
    (so the caller falls back to upright) if the rectified frame yields no usable
    regions or its vertical symmetry no longer registers."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None, [], []
    if detect_axis(np.any([r.mask for r in regions], axis=0)) is None:
        return None, [], []
    affine = _rectify_affine(rho, w0, h0, rw, rh)
    if opt.flatten:
        body, defs, cands, frame_axes = _render_body(rw, rh, regions, opt, bake=affine, rgb=rot)
        doc = render_svg_doc(w0, h0, body, defs)
    else:
        body, defs, cands, frame_axes = _render_body(rw, rh, regions, opt, rgb=rot)
        wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
                f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
        doc = render_svg_doc(w0, h0, [wrap, *body, "</g>"], defs)
    axes = [_map_axis(a, affine) for a in frame_axes]
    return doc, cands, axes


def _flatten_on_white(im: Image.Image) -> np.ndarray:
    """RGB (H,W,3) uint8 with any alpha composited onto WHITE (a transparent surround is
    background, not a mark). PIL's plain `convert("RGB")` instead DROPS alpha, keeping each
    pixel's stored RGB — so transparent icon backgrounds (typically stored black) become a
    black region, semi-transparent edges keep over-saturated colours, and a spurious white
    anti-aliasing ring appears: mangling the most common logo input."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def idealize(image, *, options: Options | None = None, report: bool = False) -> str | tuple[str, IdealizeReport]:
    """Idealize a raster mark into SVG. With `report=True`, returns
    `(svg, IdealizeReport)`; otherwise returns the SVG string (back-compatible)."""
    opt = options or Options()
    if isinstance(image, str):
        with Image.open(image) as im:
            arr = _flatten_on_white(im)
    else:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:            # RGBA array -> composite on white
            arr = _flatten_on_white(Image.fromarray(arr, "RGBA"))
    h0, w0 = arr.shape[:2]

    w, h, regions = _segment_image(arr, opt)
    if not regions:
        svg, cands, axes = render_svg_doc(w, h, []), [], []
    else:
        svg, cands, axes = None, [], []
        # Any-axis symmetry: rectify a tilted mirror upright, idealize there, wrap back.
        if not opt.no_symmetry:
            silhouette = np.any([r.mask for r in regions], axis=0)
            if detect_axis(silhouette) is None:
                rho = detect_symmetry_rotation(silhouette)
                if rho is not None:
                    rectified, rcands, raxes = _idealize_rectified(arr, opt, rho, w0, h0)
                    if rectified is not None:
                        svg, cands, axes = rectified, rcands, raxes
        if svg is None:
            body, defs, cands, axes = _render_body(w, h, regions, opt, rgb=arr)
            svg = render_svg_doc(w, h, body, defs)

    return (svg, _build_report(cands, axes)) if report else svg
