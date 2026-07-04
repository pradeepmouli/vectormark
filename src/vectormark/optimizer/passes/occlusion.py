from __future__ import annotations

import numpy as np

from ...candidate import FlatFill
from ...fit import Shape
from ...occlusion import ScenePrimitive, reconstruct_scene, region_adjacency
from ...symmetry import detect_axis
from ...types import Region
from ..framework import Proposal
from ..gate import rasterize
from ..vector_region import VectorRegion, to_polygon


def _flat_hex(region: VectorRegion) -> str | None:
    if isinstance(region.fill, FlatFill):
        return region.fill.hex
    return region.color_hex


def _as_trace_region(region: VectorRegion) -> Region | None:
    color_hex = _flat_hex(region)
    if color_hex is None or region.raster.size == 0:
        return None
    return Region(
        label=int(region.id),
        mask=np.asarray(region.raster, dtype=bool).copy(),
        color_hex=color_hex,
        coverage=region.coverage,
    )


def _shape_from_scene_primitive(primitive: ScenePrimitive) -> Shape:
    return Shape(primitive.kind, dict(primitive.params))


def _child_from_scene_item(
    item: ScenePrimitive | Shape,
    *,
    id: int,
    shape_hw: tuple[int, int],
    z_offset: float = 0.0,
) -> VectorRegion:
    if isinstance(item, ScenePrimitive):
        shape = _shape_from_scene_primitive(item)
        fill = FlatFill(item.color_hex)
        z = z_offset + float(item.z)
        raster = rasterize(to_polygon(shape), shape_hw)
        color_hex = item.color_hex
    else:
        shape = item
        color_hex = str(item.params["color_hex"])
        fill = FlatFill(color_hex)
        z = z_offset + float(item.params.get("z", 0))
        raster = rasterize(to_polygon(shape), shape_hw)

    return VectorRegion(
        id=id,
        current=shape,
        fill=fill,
        z=z,
        raster=raster,
        footprint=to_polygon(shape),
        color_hex=color_hex,
        diagnostics={"occlusion": {"reconstructed_child": True}},
    )


def occlusion_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    no_symmetry: bool = False,
) -> list[Proposal]:
    leaves = [obj for obj in sorted(objects, key=lambda current: int(current.id)) if obj.is_leaf]
    trace_regions = [_as_trace_region(obj) for obj in leaves]
    region_by_id = {region.label: region for region in trace_regions if region is not None}
    if len(region_by_id) < 2:
        return []

    shape_hw = next(iter(masks.values())).shape if masks else next(iter(region_by_id.values())).mask.shape
    silhouette = np.zeros(shape_hw, dtype=bool)
    for region in region_by_id.values():
        if region.mask.shape == shape_hw:
            silhouette |= region.mask
    axis = None if no_symmetry else detect_axis(silhouette)
    adjacency = region_adjacency(list(region_by_id.values()))
    seen: set[int] = set()
    proposals: list[Proposal] = []
    next_id = max([int(obj.id) for obj in objects], default=-1) + 1

    for root_id in sorted(region_by_id):
        if root_id in seen:
            continue
        component_ids: set[int] = set()
        stack = [root_id]
        while stack:
            obj_id = stack.pop()
            if obj_id in component_ids:
                continue
            component_ids.add(obj_id)
            stack.extend(adjacency[obj_id] - component_ids)
        seen.update(component_ids)
        if len(component_ids) < 2:
            continue

        component_regions = [region_by_id[obj_id] for obj_id in sorted(component_ids)]
        reconstructed, remaining = reconstruct_scene(component_regions, axis, shape_hw)
        if not reconstructed:
            continue
        remaining_ids = {int(region.label) for region in remaining}
        consumed_ids = tuple(sorted(component_ids - remaining_ids))
        if len(consumed_ids) < 2:
            continue

        base_z = min(float(obj.z) for obj in leaves if int(obj.id) in consumed_ids)
        children: list[VectorRegion] = []
        primitive_id_index = 0
        for item in reconstructed:
            if isinstance(item, ScenePrimitive) and primitive_id_index < len(consumed_ids):
                child_id = consumed_ids[primitive_id_index]
                primitive_id_index += 1
            else:
                child_id = next_id
                next_id += 1
            children.append(_child_from_scene_item(item, id=child_id, shape_hw=shape_hw, z_offset=base_z))

        branch = VectorRegion.branch(
            id=consumed_ids[0],
            children=children,
            z=base_z,
            raster=np.logical_or.reduce([region_by_id[obj_id].mask for obj_id in consumed_ids]),
            diagnostics={
                "occlusion": {
                    "accepted": True,
                    "consumed_ids": [int(obj_id) for obj_id in consumed_ids],
                    "children": len(children),
                }
            },
        )
        proposals.append(Proposal(consumed_ids, [branch]))

    return proposals
