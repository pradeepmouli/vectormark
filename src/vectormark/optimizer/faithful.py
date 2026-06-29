from __future__ import annotations

import numpy as np

from ..candidate import FlatFill
from ..components import decompose_components
from ..contour import region_contours
from ..fill_fit import fit_fill
from ..fit import Shape, fit_path
from ..pipeline import Options, _segment_image
from ..surface_merge import merge_surfaces
from ..types import Region
from .optobject import OptObject


def _contour_area(contour: np.ndarray) -> float:
    if len(contour) < 3:
        return 0.0
    x = contour[:, 0]
    y = contour[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _significant_contours(region: Region, opt: Options) -> list[np.ndarray]:
    contours = [c for c in region_contours(region.mask, coverage=region.coverage) if len(c) >= 3]
    if len(contours) <= 1:
        return contours

    largest = _contour_area(contours[0])
    if largest <= 0.0:
        return contours[:1]

    min_area = max(1.0, largest * opt.min_region_fraction)
    kept = [c for c in contours if _contour_area(c) >= min_area]
    return kept or contours[:1]


def _faithful_shape(region: Region, opt: Options) -> Shape | None:
    contours = _significant_contours(region, opt)
    if not contours:
        return None

    ds = [
        fit_path(
            contour,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=opt.cubic_paths,
        ).params["d"]
        for contour in contours
    ]
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
