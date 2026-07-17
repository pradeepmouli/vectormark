from __future__ import annotations

import numpy as np

from ...skia_geometry import SkPath
from ...candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill
from ...fit import Shape
from ..framework import Proposal
from ..gate import rasterize
from ..vector_region import VectorRegion, _parse_subpaths, _ring_area, _sample_subpath, to_polygon

_MIN_SUBPATH_AREA_FRACTION = 0.01
_SOURCE_Z_OFFSET = 0.0
_COVER_Z_OFFSET = 0.4
_FILL_COVERAGE_THRESHOLD = 0.5
_DEFAULT_BACKGROUND_FILL = FlatFill("#FFFFFF")


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


def _source_coverage(child_raster: np.ndarray, source_mask: np.ndarray | None) -> float:
    if source_mask is None:
        return 1.0
    child_area = int(np.count_nonzero(child_raster))
    if child_area == 0:
        return 0.0
    return float(np.count_nonzero(child_raster & source_mask) / child_area)


def _fill_from_render_evidence(
    child_raster: np.ndarray,
    child_footprint: SkPath,
    *,
    source: VectorRegion,
    source_mask: np.ndarray | None,
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> tuple[Fill | None, float, int | None]:
    source_coverage = _source_coverage(child_raster, source_mask)
    if source_coverage >= _FILL_COVERAGE_THRESHOLD:
        assert source.fill is not None
        return source.fill, source_coverage, source.id

    child_area = int(np.count_nonzero(child_raster))
    best_fill: Fill | None = None
    best_coverage = 0.0
    best_id: int | None = None
    best_z = float("-inf")
    if child_area == 0:
        return best_fill, best_coverage, best_id

    for candidate in objects:
        if candidate.id == source.id or candidate.fill is None:
            continue
        candidate_mask = masks.get(candidate.id)
        if candidate_mask is None:
            candidate_mask = candidate.raster
        if candidate_mask.shape != child_raster.shape:
            continue
        coverage = float(np.count_nonzero(child_raster & candidate_mask) / child_area)
        if coverage > best_coverage or (coverage == best_coverage and float(candidate.z) > best_z):
            best_fill = candidate.fill
            best_coverage = coverage
            best_id = candidate.id
            best_z = float(candidate.z)

    if best_coverage >= _FILL_COVERAGE_THRESHOLD:
        return best_fill, best_coverage, best_id
    if isinstance(source.fill, FlatFill):
        # A nested winding subpath with no visible source layer is a real
        # cutout.  Materialize the canvas plate so outer and inner geometry can
        # be optimized independently instead of retaining one compound path.
        return _DEFAULT_BACKGROUND_FILL, best_coverage, None
    if not isinstance(source.fill, (LinearGradientFill, RadialGradientFill)):
        return None, best_coverage, best_id
    return _DEFAULT_BACKGROUND_FILL, best_coverage, None


def split_compound_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    proposals: list[Proposal] = []
    next_id = max((int(obj.id) for obj in objects), default=-1) + 1
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        if not isinstance(obj.fill, (FlatFill, LinearGradientFill, RadialGradientFill)):
            continue

        subpaths = _parse_subpaths(str(obj.current.params.get("d", "")))
        if len(subpaths) < 2:
            continue

        areas = [abs(_subpath_area(tokens)) for tokens in subpaths]
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
        source_subpath_index = max(kept, key=lambda item: item[2])[0]
        kept = sorted(kept, key=lambda item: (item[0] != source_subpath_index, item[0]))

        source_mask = masks.get(obj.id)
        if source_mask is None:
            source_mask = obj.raster
        shape_hw = source_mask.shape if source_mask is not None else None

        footprints: list[SkPath] = []
        for _subpath_index, tokens, _area in kept:
            footprint = to_polygon(_subpath_shape(tokens))
            if footprint.is_empty:
                break
            footprints.append(footprint)
        if len(footprints) != len(kept):
            continue

        # SVG permits a nested hole to rely on winding direction instead of an
        # explicit fill-rule.  Treat that as a compound cutout too: retaining
        # it as one path prevents primitive fitting of the exterior and the
        # interior independently.  Disconnected subpaths without a fill rule
        # remain one path, because they may be intentional islands rather than
        # a layer-plus-cutout relationship.
        if not obj.current.params.get("fill_rule"):
            source_footprint = footprints[0]
            contains_nested_subpath = any(
                source_footprint._path.contains(float(footprint.centroid.x), float(footprint.centroid.y))
                for footprint in footprints[1:]
            )
            if not contains_nested_subpath:
                continue

        children: list[VectorRegion] = []
        for output_index, (_subpath_index, tokens, _area) in enumerate(kept):
            child_id = obj.id if output_index == 0 else next_id
            if output_index != 0:
                next_id += 1
            shape = _subpath_shape(tokens)
            footprint = footprints[output_index]
            raster = rasterize(footprint, shape_hw) if shape_hw is not None else obj.raster
            if _subpath_index == source_subpath_index:
                child_fill = obj.fill
                fill_coverage = 1.0
                fill_source_id = obj.id
            else:
                child_fill, fill_coverage, fill_source_id = _fill_from_render_evidence(
                    raster,
                    footprint,
                    source=obj,
                    source_mask=source_mask,
                    objects=objects,
                    masks=masks,
            )
            if child_fill is None:
                children = []
                break
            children.append(
                VectorRegion(
                    child_id,
                    shape,
                    child_fill,
                    float(obj.z) + (_SOURCE_Z_OFFSET if output_index == 0 else _COVER_Z_OFFSET + output_index * 0.01),
                    footprint=footprint,
                    raster=raster,
                    original=shape,
                    source_label=obj.source_label,
                    color_hex=obj.color_hex,
                    source_regions=obj.source_regions,
                    coverage=obj.coverage,
                    diagnostics={
                        "compound": {
                            "source_id": int(obj.id),
                            "subpath_index": int(_subpath_index),
                            "subpaths": len(subpaths),
                            "fill_coverage": fill_coverage,
                            "fill_source_id": None if fill_source_id is None else int(fill_source_id),
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
            source_regions=obj.source_regions,
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
