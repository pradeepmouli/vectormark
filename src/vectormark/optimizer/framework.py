from __future__ import annotations

from collections import namedtuple
from typing import Protocol

import numpy as np

from .gate import BUDGET, rasterize
from .vector_region import VectorRegion

Proposal = namedtuple("Proposal", "obj_ids new_objects")


class Pass(Protocol):
    def __call__(
        self,
        objects: list[VectorRegion],
        masks: dict[int, np.ndarray],
    ) -> list[Proposal]: ...


def _proposal_key(proposal: Proposal) -> tuple[int, ...]:
    return tuple(sorted(int(obj_id) for obj_id in proposal.obj_ids))


def _object_key(obj: VectorRegion) -> tuple[float, int]:
    return (float(obj.z), int(obj.id))


def _shape_key(obj: VectorRegion) -> tuple[object, ...]:
    if obj.current is None:
        return (
            *_object_key(obj),
            "branch",
            tuple(_shape_key(child) for child in obj.leaves()),
            repr(obj.fill),
        )
    return (
        *_object_key(obj),
        obj.current.kind,
        tuple(sorted((str(k), repr(v)) for k, v in obj.current.params.items())),
        repr(obj.fill),
    )


def _proposal_sort_key(proposal: Proposal) -> tuple[tuple[int, ...], tuple[tuple[object, ...], ...]]:
    return (_proposal_key(proposal), tuple(sorted(_shape_key(obj) for obj in proposal.new_objects)))


def _union_masks(mask_by_id: dict[int, np.ndarray], obj_ids) -> np.ndarray:
    ordered_ids = tuple(sorted(int(obj_id) for obj_id in obj_ids))
    union = np.zeros_like(mask_by_id[ordered_ids[0]], dtype=bool)
    for obj_id in ordered_ids:
        union |= np.asarray(mask_by_id[obj_id], dtype=bool)
    return union


def _pass_name(pass_fn: Pass) -> str:
    name = getattr(pass_fn, "__name__", None)
    if name:
        return str(name)
    return pass_fn.__class__.__name__


def _annotate_replacement(
    replacement: VectorRegion,
    *,
    pass_name: str,
    proposal_ids: tuple[int, ...],
    raster: np.ndarray,
) -> VectorRegion:
    return replacement.with_diagnostics(
        raster=raster,
        diagnostics={
            pass_name: {
                "accepted": True,
                "proposal_ids": [int(obj_id) for obj_id in proposal_ids],
            }
        },
    )


def _masks_from_objects(objects, fallback: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for obj in objects:
        if obj.raster.size:
            out[obj.id] = np.asarray(obj.raster, dtype=bool).copy()
        elif obj.id in fallback:
            out[obj.id] = np.asarray(fallback[obj.id], dtype=bool).copy()
    return out


def _optimize_branch_children(
    objects: list[VectorRegion],
    pass_fn: Pass,
    pass_name: str,
    *,
    budget: float,
) -> list[VectorRegion]:
    updated: list[VectorRegion] = []
    for obj in objects:
        if obj.is_leaf:
            updated.append(obj)
            continue
        child_objects = list(obj.children)
        child_masks = _masks_from_objects(child_objects, {})
        optimized_children = optimize(child_objects, child_masks, [pass_fn], budget=budget)
        if len(optimized_children) == len(obj.children) and all(
            optimized is original
            for optimized, original in zip(optimized_children, obj.children, strict=True)
        ):
            updated.append(obj)
            continue
        updated.append(
            obj.with_children(
                optimized_children,
                diagnostics={
                    pass_name: {
                        "children_optimized": True,
                    }
                },
            )
        )
    return updated


def optimize(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    passes: Iterable[Pass],
    *,
    budget: float = BUDGET,
) -> list[VectorRegion]:
    current_objects = sorted(objects, key=_object_key)
    current_masks = {obj_id: np.asarray(mask, dtype=bool).copy() for obj_id, mask in masks.items()}

    for pass_fn in passes:
        pass_name = _pass_name(pass_fn)
        current_objects = _optimize_branch_children(
            current_objects,
            pass_fn,
            pass_name,
            budget=budget,
        )
        current_masks = _masks_from_objects(current_objects, current_masks)
        current_objects = sorted(current_objects, key=_object_key)
        pass_objects = list(current_objects)
        pass_masks = {obj_id: mask.copy() for obj_id, mask in current_masks.items()}
        pass_original_by_id = {obj.id: obj for obj in pass_objects}
        proposals = sorted(pass_fn(pass_objects, pass_masks), key=_proposal_sort_key)
        consumed_in_pass: set[int] = set()

        for proposal in proposals:
            raw_proposal_ids = [int(obj_id) for obj_id in proposal.obj_ids]
            if len(set(raw_proposal_ids)) != len(raw_proposal_ids):
                continue
            proposal_ids = _proposal_key(proposal)
            if not proposal_ids:
                continue
            if not proposal.new_objects:
                continue
            replacement_ids = [int(obj.id) for obj in proposal.new_objects]
            if len(set(replacement_ids)) != len(replacement_ids):
                continue
            if any(obj_id in consumed_in_pass for obj_id in proposal_ids):
                continue

            if any(obj_id not in pass_original_by_id or obj_id not in pass_masks for obj_id in proposal_ids):
                continue
            if any(
                replacement.id in pass_original_by_id and replacement.id not in proposal_ids
                for replacement in proposal.new_objects
            ):
                continue
            current_object_ids = {obj.id for obj in current_objects}
            if any(
                replacement.id in current_object_ids and replacement.id not in proposal_ids
                for replacement in proposal.new_objects
            ):
                continue

            union_mask = _union_masks(pass_masks, proposal_ids)
            split_replacement = len(proposal_ids) == 1 and len(proposal.new_objects) > 1

            preserved_masks = {
                obj_id: np.asarray(pass_masks[obj_id], dtype=bool).copy()
                for obj_id in proposal_ids
            }
            consumed_in_pass.update(proposal_ids)

            replacements = sorted(proposal.new_objects, key=_object_key)
            remaining = [obj for obj in current_objects if obj.id not in proposal_ids]

            for obj_id in proposal_ids:
                current_masks.pop(obj_id, None)

            assigned_replacement_ids: set[int] = set()
            replacement_masks: dict[int, np.ndarray] = {}
            for replacement in replacements:
                if replacement.id in current_masks:
                    current_masks.pop(replacement.id, None)

                if split_replacement:
                    replacement_masks[replacement.id] = rasterize(replacement.footprint, union_mask.shape)
                    current_masks[replacement.id] = replacement_masks[replacement.id]
                    continue

                if replacement.id in pass_original_by_id and replacement.id in proposal_ids:
                    if replacement.id not in assigned_replacement_ids:
                        replacement_masks[replacement.id] = preserved_masks[replacement.id]
                        current_masks[replacement.id] = replacement_masks[replacement.id]
                        assigned_replacement_ids.add(replacement.id)
                        continue

                replacement_masks[replacement.id] = union_mask.copy()
                current_masks[replacement.id] = replacement_masks[replacement.id]

            annotated_replacements = [
                _annotate_replacement(
                    replacement,
                    pass_name=pass_name,
                    proposal_ids=proposal_ids,
                    raster=replacement_masks[replacement.id],
                )
                for replacement in replacements
            ]
            current_objects = sorted(remaining + annotated_replacements, key=_object_key)

    return current_objects
