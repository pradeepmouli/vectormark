from __future__ import annotations

import numpy as np

from ...fit import Shape, atomic_flatten_path_d
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon


def atomic_flatten_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
) -> list[Proposal]:
    """Run final endpoint-preserving Q/L command reduction.

    This deliberately sits after symmetry and stitching.  Its transforms keep
    every segment endpoint fixed, so it cannot reopen a seam the preceding
    pass established.
    """
    del masks
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        d = str(obj.current.params.get("d", ""))
        flattened = atomic_flatten_path_d(d, epsilon=epsilon)
        if flattened == d:
            continue
        shape = Shape("path", {**obj.current.params, "d": flattened})
        proposals.append(Proposal((obj.id,), [obj.with_current(shape, footprint=to_polygon(shape))]))
    return proposals
