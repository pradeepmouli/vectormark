"""Top-level orchestration: raster path/array -> structured SVG string."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .color import extract_palette, quantize
from .contour import region_contours
from .emit import mirror_use, path_svg, reflect_path_d, render_svg_doc, shape_to_path_d, shape_to_svg
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .segment import segment
from .symmetry import classify_regions, detect_axis
from .types import Axis, Region


@dataclass
class Options:
    epsilon: float = 1.5          # primitive/polygon recognition tolerance (px)
    max_error: float = 1.0        # Bézier fit tolerance (px)
    max_colors: int = 16
    flatten: bool = False
    no_symmetry: bool = False


def _fit_region(region: Region, opt: Options, axis: Axis | None) -> Shape | None:
    contours = [c for c in region_contours(region.mask) if len(c) >= 3]
    if not contours:
        return None
    if len(contours) == 1:
        contour = contours[0]
        shape = recognize_primitive(contour, epsilon=opt.epsilon)
        if shape is None:
            shape = recognize_polygon(contour, epsilon=opt.epsilon)
        if shape is None:
            shape = fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error)
    else:
        # holes / counters: outer + inner contours as subpaths, even-odd fill.
        # Skip primitive/polygon recognition (those see the outer ring only and
        # would fill the hole solid).
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error).params["d"]
            for c in contours
        )
        shape = Shape("path", {"d": d, "fill_rule": "evenodd"})
    if axis is not None:
        shape = _snap_to_axis(shape, axis)
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
    regions = segment(q)

    if not regions:
        return render_svg_doc(w, h, [])

    silhouette = np.any([r.mask for r in regions], axis=0)
    axis = None if opt.no_symmetry else detect_axis(silhouette)

    straddlers: list[Region]
    pairs: list[tuple[Region, Region]]
    if axis is not None:
        straddlers, pairs = classify_regions(regions, axis)
    else:
        straddlers, pairs = list(regions), []

    # back-to-front: paint larger regions first
    body: list[str] = []
    drawn = [(r, False) for r in straddlers] + [(canon, True) for canon, _ in pairs]
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    eid = 0
    for region, is_pair in drawn:
        shape = _fit_region(region, opt, axis if not is_pair else None)
        if shape is None:
            continue
        if opt.flatten:
            # everything becomes a <path>; the mirror is baked as a reflected
            # path (no <use>, no basic shapes) for maximum portability
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
