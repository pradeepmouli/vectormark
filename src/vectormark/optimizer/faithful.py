from __future__ import annotations

import numpy as np

from ..candidate import FlatFill
from ..components import decompose_components
from ..contour import region_contours
from ..fill_fit import fit_fill
from ..fit import Shape, _fmt, fit_path
from ..pipeline import Options, _segment_image
from ..surface_merge import merge_surfaces
from ..types import Region
from .optobject import OptObject


def _region_path_contours(region: Region) -> list[np.ndarray]:
    return [c for c in region_contours(region.mask, coverage=region.coverage) if len(c) >= 3]


def _contour_line_path(contour: np.ndarray) -> str:
    pts = np.asarray(contour, dtype=float)
    if len(pts) == 0:
        return ""
    d = f"M{_fmt(pts[0][0])} {_fmt(pts[0][1])} "
    for x, y in pts[1:]:
        d += f"L{_fmt(x)} {_fmt(y)} "
    return d + "Z"


def _faithful_shape(region: Region, opt: Options) -> Shape | None:
    contours = _region_path_contours(region)
    if not contours:
        return None

    ds: list[str] = []
    for contour in contours:
        fitted = fit_path(
            contour,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=opt.cubic_paths,
        )
        ds.append(fitted.params["d"] if fitted is not None else _contour_line_path(contour))
    params = {"d": " ".join(ds)}
    if len(ds) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def faithful_objects(arr: np.ndarray, opt: Options) -> tuple[list[OptObject], dict[int, np.ndarray]]:
    w, h, regions = _segment_image(arr, opt)
    components = decompose_components(regions, (h, w))

    merged_regions: list[Region] = []
    fills_by_label = {}
    for comp in components:
        flat_filled = [(r, FlatFill(r.color_hex)) for r in comp]
        merged = merge_surfaces(flat_filled, arr)
        comp_regions = [r for r, _ in merged]
        merged_regions.extend(comp_regions)
        fills_by_label.update({
            r.label: fit_fill(r.mask, arr, flat_hex=r.color_hex)
            for r in comp_regions
        })

    ordered = sorted(
        merged_regions,
        key=lambda r: (-r.area, r.label, r.color_hex),
    )

    objects: list[OptObject] = []
    true_masks: dict[int, np.ndarray] = {}
    for idx, region in enumerate(ordered):
        shape = _faithful_shape(region, opt)
        if shape is None:
            continue
        obj = OptObject(
            id=idx,
            exact=shape,
            fill=fills_by_label.get(region.label, FlatFill(region.color_hex)),
            z=idx,
        )
        objects.append(obj)
        true_masks[obj.id] = region.mask.copy()

    return objects, true_masks
