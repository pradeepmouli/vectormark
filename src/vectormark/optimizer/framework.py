from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from .gate import BUDGET, gate_ok
from .optobject import OptObject

Proposal = namedtuple("Proposal", "obj_ids new_objects")


class Pass(Protocol):
    def __call__(
        self,
        objects: list[OptObject],
        masks: dict[int, np.ndarray],
    ) -> list[Proposal]: ...


def _proposal_key(proposal: Proposal) -> tuple[int, ...]:
    return tuple(sorted(int(obj_id) for obj_id in proposal.obj_ids))


def _object_key(obj: OptObject) -> tuple[int, int]:
    return (int(obj.id), int(obj.z))


def _shape_key(obj: OptObject) -> tuple[object, ...]:
    return (
        *_object_key(obj),
        obj.exact.kind,
        tuple(sorted((str(k), repr(v)) for k, v in obj.exact.params.items())),
        repr(obj.fill),
    )


def _proposal_sort_key(proposal: Proposal) -> tuple[tuple[int, ...], tuple[tuple[object, ...], ...]]:
    return (_proposal_key(proposal), tuple(sorted(_shape_key(obj) for obj in proposal.new_objects)))


def _geometry_changed(original: OptObject, replacement: OptObject) -> bool:
    return original.exact != replacement.exact


def _union_masks(mask_by_id: dict[int, np.ndarray], obj_ids: Iterable[int]) -> np.ndarray:
    ordered_ids = tuple(sorted(int(obj_id) for obj_id in obj_ids))
    union = np.zeros_like(mask_by_id[ordered_ids[0]], dtype=bool)
    for obj_id in ordered_ids:
        union |= np.asarray(mask_by_id[obj_id], dtype=bool)
    return union


def _matched_original(
    proposal_ids: tuple[int, ...],
    originals: dict[int, OptObject],
    replacement: OptObject,
) -> OptObject | None:
    if replacement.id in proposal_ids and replacement.id in originals:
        return originals[replacement.id]
    if len(proposal_ids) == 1:
        return originals[proposal_ids[0]]
    return None


def _gate_mask(
    proposal_ids: tuple[int, ...],
    originals: dict[int, OptObject],
    current_masks: dict[int, np.ndarray],
    replacement: OptObject,
    union_mask: np.ndarray,
) -> np.ndarray | None:
    original = _matched_original(proposal_ids, originals, replacement)
    if original is None:
        if len(proposal_ids) == 1:
            return current_masks[proposal_ids[0]]
        return union_mask
    if not _geometry_changed(original, replacement):
        return None
    return current_masks[original.id]


def optimize(
    objects: list[OptObject],
    masks: dict[int, np.ndarray],
    passes: Iterable[Pass],
    *,
    budget: float = BUDGET,
) -> list[OptObject]:
    current_objects = sorted(objects, key=_object_key)
    current_masks = {obj_id: np.asarray(mask, dtype=bool).copy() for obj_id, mask in masks.items()}

    for pass_fn in passes:
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

            union_mask = _union_masks(pass_masks, proposal_ids)
            gates_ok = True
            for replacement in proposal.new_objects:
                gate_mask = _gate_mask(
                    proposal_ids,
                    pass_original_by_id,
                    pass_masks,
                    replacement,
                    union_mask,
                )
                if gate_mask is None:
                    continue
                if not gate_ok(replacement.flat, gate_mask, budget=budget):
                    gates_ok = False
                    break

            if not gates_ok:
                continue

            preserved_masks = {
                obj_id: np.asarray(pass_masks[obj_id], dtype=bool).copy()
                for obj_id in proposal_ids
            }
            consumed_in_pass.update(proposal_ids)

            remaining = [obj for obj in current_objects if obj.id not in proposal_ids]
            replacements = sorted(proposal.new_objects, key=_object_key)
            current_objects = sorted(remaining + replacements, key=_object_key)

            for obj_id in proposal_ids:
                current_masks.pop(obj_id, None)

            assigned_replacement_ids: set[int] = set()
            for replacement in replacements:
                if replacement.id in current_masks:
                    current_masks.pop(replacement.id, None)

                if replacement.id in pass_original_by_id and replacement.id in proposal_ids:
                    if replacement.id not in assigned_replacement_ids:
                        current_masks[replacement.id] = preserved_masks[replacement.id]
                        assigned_replacement_ids.add(replacement.id)
                        continue

                current_masks[replacement.id] = union_mask.copy()

    return current_objects
