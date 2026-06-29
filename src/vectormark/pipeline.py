"""Top-level orchestration: raster path/array -> structured SVG string."""
from __future__ import annotations

import re
import types
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .color import extract_palette, quantize
from .contour import region_corner_radius
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
from .candidate import Candidate, Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .components import decompose_components
from .fill_fit import fit_fill
from .fit import Shape, _fmt
from .occlusion import ScenePrimitive, reconstruct_scene
from .surface_merge import merge_surfaces
from .segment import segment
from .selection import SelectionPolicy
from .selector import select_geometry
from .symmetry import classify_regions, detect_axis, detect_symmetry_rotation
from .types import Axis, Region


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
    sym_tol: float = 0.10        # mirror-axis detection tolerance (reflection mismatch)
    straddle_iou: float = 0.96   # min self-reflection IoU to force a SINGLE region symmetric
    pair_iou: float = 0.90       # min IoU to treat two regions as a mirror pair
    working_max_dim: int | None = 768  # downscale inputs whose longest side exceeds this
                                       # (LANCZOS) before segmentation; None disables.


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
    emitted element count, the detected mirror axes (one segment per component
    with a vertical mirror, in output-frame coords), and per-region symmetry
    diagnostics. Diagnostic annotation."""

    strategies: Mapping[str, int]
    gradients: int
    elements: int
    axes: tuple[AxisLine, ...]
    symmetry: tuple  # per-region (label, self_iou, decision) entries from classify_regions

    @staticmethod
    def empty() -> "IdealizeReport":
        return IdealizeReport(types.MappingProxyType({}), 0, 0, (), ())


def _map_axis(a: AxisLine, affine: Affine) -> AxisLine:
    x1, y1 = apply_affine_point(affine, a.x1, a.y1)
    x2, y2 = apply_affine_point(affine, a.x2, a.y2)
    return AxisLine(x1, y1, x2, y2)


def _build_report(cands: list[Candidate], axes: list[AxisLine], sym_diags: list | None = None) -> IdealizeReport:
    strategies: dict[str, int] = {}
    gradients = 0
    for c in cands:
        if isinstance(c.fill, (LinearGradientFill, RadialGradientFill)):
            gradients += 1
        if c.strategy is not None:                 # None for occlusion / lens
            strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    return IdealizeReport(
        types.MappingProxyType(dict(strategies)),
        gradients,
        len(cands),
        tuple(axes),
        tuple(sym_diags) if sym_diags is not None else (),
    )


def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
    loners: list[Region], fills: dict[int, Fill],
    opt: Options, axis: Axis | None,
    source_rgb: np.ndarray | None, *, base: int = 0,
) -> list[Candidate]:
    """Decide geometry + fill per element and return the candidate list in exact
    paint order: occlusion (by z) -> regions (by area desc). Elements whose geometry
    fit returns None are dropped (matching the old per-loop `continue`, so the
    emit-time id sequence is unchanged). A region absent from `fills` falls back to
    FlatFill(region.color_hex)."""
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
        cr = opt.corner_radius if opt.corner_radius is not None else region_corner_radius(region.mask)
        shape, strategy = select_geometry(region, opt, fit_axis, cr, source_rgb,
                                          element=element, eid=eid)
        if shape is None:
            continue
        fill = fills.get(region.label, FlatFill(region.color_hex))
        cands.append(Candidate(shape, fill, "region",
                               mirror=axis if is_pair else None, strategy=strategy))

    return cands


def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str], list[Candidate], list[AxisLine], list]:
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
    sym_diags: list = []
    for comp in components:
        silhouette = np.any([r.mask for r in comp], axis=0)
        axis = None if opt.no_symmetry else detect_axis(silhouette, tol=opt.sym_tol)
        if axis is not None:
            ys = np.nonzero(silhouette)[0]
            frame_axes.append(AxisLine(float(axis.x), float(ys.min()), float(axis.x), float(ys.max())))
        reconstructed, comp = reconstruct_scene(comp, axis, (h, w))

        # Seam-merge adjacent regions into surfaces, then fit fill per merged surface.
        # This 2-pass drives only path A (seam_is_soft): flat fills are passed into
        # merge_surfaces so none of the surfaces entering the merge loop carry gradient
        # fills, and gradients_continuous (path B) therefore returns False for every pair.
        # Path B is intentionally preserved in merge_surfaces and is covered by its own
        # unit tests, but is NOT exercised end-to-end from here because intermediate
        # gradient-type mismatches (radial vs linear on partial merges) made per-region
        # path-B merges unreliable.
        # TODO(follow-up): revisit B-primary once per-region gradient-kind is stable.
        if rgb is not None:
            flat_filled = [(r, FlatFill(r.color_hex)) for r in comp]
            merged = merge_surfaces(flat_filled, rgb)
            comp = [r for r, _ in merged]
            fills = {r.label: fit_fill(r.mask, rgb, flat_hex=r.color_hex) for r, _ in merged}
        else:
            fills = {}

        if axis is not None:
            straddlers, pairs, loners = classify_regions(
                comp, axis,
                straddle_iou=opt.straddle_iou,
                pair_iou=opt.pair_iou,
                diagnostics=sym_diags,
            )
        else:
            straddlers, pairs, loners = list(comp), [], []

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, fills, opt, axis, rgb,
            base=len(cands),
        )

    def emit(d: str, fill: str, rule: str | None = None) -> str:
        return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)

    def _fill_attr(fill: Fill) -> str:
        if isinstance(fill, RasterFill):
            # userSpaceOnUse pattern: map to the baked frame via patternTransform
            # (bake is set only in flatten mode; None otherwise -> absolute coords).
            return resolve_fill(fill, defs, transform=bake)
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

    return body, defs, cands, frame_axes, sym_diags


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


def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> tuple[str | None, list[Candidate], list[AxisLine], list]:
    """Rotate the image so the tilted mirror axis is vertical and idealize there.
    Non-flatten output keeps the symmetry (`<use>` mirror about the vertical axis)
    and is wrapped in one inverse-rotation `<g>`; flatten output bakes that same
    rotation into the path coordinates so no transform survives. Returns (None, [], [], [])
    (so the caller falls back to upright) if the rectified frame yields no usable
    regions or its vertical symmetry no longer registers."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None, [], [], []
    if detect_axis(np.any([r.mask for r in regions], axis=0), tol=opt.sym_tol) is None:
        return None, [], [], []
    affine = _rectify_affine(rho, w0, h0, rw, rh)
    bake = affine if opt.flatten else None
    body, defs, cands, frame_axes, sym_diags = _render_body(rw, rh, regions, opt, bake=bake, rgb=rot)
    # Bail unless the rectified frame actually exploited mirror symmetry (some region
    # classified as a straddler or pair). A tilted silhouette can register a vertical
    # axis yet have every region fall below the straddle gate (e.g. telegram's lone
    # paper-plane at IoU 0.917 < 0.96, classified a loner): then the rotated re-fit only
    # adds resampling churn versus the upright idealization, so fall back to upright
    # instead of committing a distorted result.
    if not any(decision in ("straddler", "pair") for _, _, decision in sym_diags):
        return None, [], [], []
    if opt.flatten:
        doc = render_svg_doc(w0, h0, body, defs)
    else:
        wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
                f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
        doc = render_svg_doc(w0, h0, [wrap, *body, "</g>"], defs)
    axes = [_map_axis(a, affine) for a in frame_axes]
    return doc, cands, axes, sym_diags


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


def _condition_input(arr: np.ndarray, working_max_dim: int | None) -> np.ndarray:
    """Downscale an oversized RGB array to a working resolution before segmentation, so
    input noise stops fragmenting at high pixel counts. Longest side -> working_max_dim,
    aspect-preserving, LANCZOS. Returns arr unchanged when disabled or already small.
    Downscale only — never upscale, never denoise."""
    if working_max_dim is None:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) <= working_max_dim:
        return arr
    scale = working_max_dim / max(h, w)
    img = Image.fromarray(arr).resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _set_svg_output_size(svg: str, width: int, height: int) -> str:
    """Rewrite the <svg> width/height to the original input size, leaving viewBox (working
    space) intact — a pure display scale (SVG is resolution-free)."""
    return re.sub(r'(<svg\b[^>]*?)\bwidth="\d+"\s+height="\d+"',
                  rf'\1width="{width}" height="{height}"', svg, count=1)


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
    orig_h, orig_w = arr.shape[:2]
    arr = _condition_input(arr, opt.working_max_dim)
    h0, w0 = arr.shape[:2]

    w, h, regions = _segment_image(arr, opt)
    if not regions:
        svg, cands, axes, sym_diags = render_svg_doc(w, h, []), [], [], []
    else:
        svg, cands, axes, sym_diags = None, [], [], []
        # Any-axis symmetry: rectify a tilted mirror upright, idealize there, wrap back.
        if not opt.no_symmetry:
            silhouette = np.any([r.mask for r in regions], axis=0)
            if detect_axis(silhouette, tol=opt.sym_tol) is None:
                rho = detect_symmetry_rotation(silhouette, tol=opt.sym_tol)
                if rho is not None:
                    rectified, rcands, raxes, rsym_diags = _idealize_rectified(arr, opt, rho, w0, h0)
                    if rectified is not None:
                        svg, cands, axes, sym_diags = rectified, rcands, raxes, rsym_diags
        if svg is None:
            body, defs, cands, axes, sym_diags = _render_body(w, h, regions, opt, rgb=arr)
            svg = render_svg_doc(w, h, body, defs)

    if (arr.shape[1], arr.shape[0]) != (orig_w, orig_h):
        svg = _set_svg_output_size(svg, orig_w, orig_h)
    return (svg, _build_report(cands, axes, sym_diags)) if report else svg
