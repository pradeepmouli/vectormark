from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from .gate import BUDGET, coverage_residual
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
    if replacement.id in originals:
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
    current_objects = list(objects)
    current_masks = {obj_id: np.asarray(mask, dtype=bool).copy() for obj_id, mask in masks.items()}

    for pass_fn in passes:
        proposals = sorted(pass_fn(current_objects, current_masks), key=_proposal_key)
        consumed_in_pass: set[int] = set()

        for proposal in proposals:
            proposal_ids = _proposal_key(proposal)
            if not proposal_ids:
                continue
            if any(obj_id in consumed_in_pass for obj_id in proposal_ids):
                continue

            original_by_id = {obj.id: obj for obj in current_objects}
            if any(obj_id not in original_by_id or obj_id not in current_masks for obj_id in proposal_ids):
                continue

            union_mask = _union_masks(current_masks, proposal_ids)
            gates_ok = True
            for replacement in proposal.new_objects:
                gate_mask = _gate_mask(
                    proposal_ids,
                    original_by_id,
                    current_masks,
                    replacement,
                    union_mask,
                )
                if gate_mask is None:
                    continue
                if coverage_residual(replacement.flat, gate_mask) > budget:
                    gates_ok = False
                    break

            if not gates_ok:
                continue

            preserved_masks = {
                obj_id: np.asarray(current_masks[obj_id], dtype=bool).copy()
                for obj_id in proposal_ids
            }
            consumed_in_pass.update(proposal_ids)

            insert_at = min(
                idx for idx, obj in enumerate(current_objects) if obj.id in proposal_ids
            )
            remaining = [obj for obj in current_objects if obj.id not in proposal_ids]
            current_objects = (
                remaining[:insert_at] + list(proposal.new_objects) + remaining[insert_at:]
            )

            for obj_id in proposal_ids:
                current_masks.pop(obj_id, None)

            assigned_replacement_ids: set[int] = set()
            for replacement in proposal.new_objects:
                if replacement.id in current_masks:
                    current_masks.pop(replacement.id, None)

                if replacement.id in original_by_id and replacement.id in proposal_ids:
                    if replacement.id not in assigned_replacement_ids:
                        current_masks[replacement.id] = preserved_masks[replacement.id]
                        assigned_replacement_ids.add(replacement.id)
                        continue

                current_masks[replacement.id] = union_mask.copy()

    return current_objects
