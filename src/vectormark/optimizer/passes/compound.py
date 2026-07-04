from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from ...candidate import FlatFill, LinearGradientFill
from ...fit import Shape
from ..framework import Proposal
from ..gate import rasterize
from ..vector_region import VectorRegion, _parse_subpaths, _ring_area, _sample_subpath, to_polygon

_MIN_SUBPATH_AREA_FRACTION = 0.01
_SOURCE_Z_OFFSET = 0.0
_COVER_Z_OFFSET = 0.4


def _subpath_shape(tokens: list[tuple[str, list[float]]]) -> Shape:
    parts: list[str] = []
    for command, values in tokens:
        if values:
            parts.append(f"{command}{' '.join(_fmt_value(value) for value in values)}")
        else:
            parts.append(command)
    return Shape("path", {"d": " ".join(parts)})


def _fmt_value(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _subpath_area(tokens: list[tuple[str, list[float]]]) -> float:
    points = _sample_subpath(tokens, samples=24)
    if len(points) < 3:
        return 0.0
    return float(_ring_area(points))


def split_compound_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    proposals: list[Proposal] = []
    next_id = max((int(obj.id) for obj in objects), default=-1) + 1
    shape_hw = next(iter(masks.values())).shape if masks else None

    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        if not obj.current.params.get("fill_rule"):
            continue
        if not isinstance(obj.fill, (FlatFill, LinearGradientFill)):
            continue

        subpaths = _parse_subpaths(str(obj.current.params.get("d", "")))
        if len(subpaths) < 2:
            continue

        areas = [_subpath_area(tokens) for tokens in subpaths]
        max_area = max(areas, default=0.0)
        if max_area <= 0.0:
            continue
        kept = [
            (index, tokens, area)
            for index, (tokens, area) in enumerate(zip(subpaths, areas, strict=True))
            if area >= max_area * _MIN_SUBPATH_AREA_FRACTION
        ]
        if len(kept) < 2:
            continue

        children: list[VectorRegion] = []
        for output_index, (_subpath_index, tokens, _area) in enumerate(kept):
            child_id = obj.id if output_index == 0 else next_id
            if output_index != 0:
                next_id += 1
            shape = _subpath_shape(tokens)
            footprint = to_polygon(shape)
            if isinstance(footprint, Polygon) and footprint.is_empty:
                continue
            raster = rasterize(footprint, shape_hw) if shape_hw is not None else obj.raster
            children.append(
                VectorRegion(
                    child_id,
                    shape,
                    obj.fill if output_index == 0 else FlatFill("#FFFFFF"),
                    float(obj.z) + (_SOURCE_Z_OFFSET if output_index == 0 else _COVER_Z_OFFSET + output_index * 0.01),
                    footprint=footprint,
                    raster=raster,
                    original=shape,
                    source_label=obj.source_label,
                    color_hex=obj.color_hex,
                    coverage=obj.coverage,
                    diagnostics={
                        "compound": {
                            "source_id": int(obj.id),
                            "subpath_index": int(_subpath_index),
                            "subpaths": len(subpaths),
                        }
                    },
                )
            )

        if len(children) < 2:
            continue
        branch = VectorRegion.branch(
            id=obj.id,
            children=children,
            z=obj.z,
            raster=obj.raster,
            footprint=obj.footprint,
            fill=obj.fill,
            source_label=obj.source_label,
            color_hex=obj.color_hex,
            diagnostics={
                "compound": {
                    "accepted": True,
                    "children": len(children),
                    "dropped_subpaths": len(subpaths) - len(children),
                }
            },
        )
        proposals.append(Proposal((obj.id,), [branch]))

    return proposals
