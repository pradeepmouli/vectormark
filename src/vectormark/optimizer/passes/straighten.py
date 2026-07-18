from __future__ import annotations

import numpy as np

from ...candidate import FlatFill
from ...fit import Shape
from ..framework import Proposal
from ..vector_region import VectorRegion, _parse_subpaths, to_polygon
from .simplify import _normalized_subpath_d


def _straightened_shape(shape: Shape, *, epsilon: float) -> Shape | None:
    """Replace only curve runs already within ``epsilon`` of their chord."""
    if shape.kind != "path":
        return None
    d = str(shape.params.get("d", ""))
    subpaths = _parse_subpaths(d)
    if not subpaths:
        return None
    normalized = " ".join(_normalized_subpath_d(tokens, epsilon=epsilon) for tokens in subpaths)
    if normalized == d:
        return None
    return Shape("path", {**shape.params, "d": normalized})


def straighten_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
) -> list[Proposal]:
    """Normalize nearly straight Bézier segments to lines in linear time."""
    del masks
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        if not isinstance(obj.fill, FlatFill):
            continue
        shape = _straightened_shape(obj.current, epsilon=epsilon)
        if shape is not None:
            proposals.append(Proposal((obj.id,), [obj.with_current(shape, footprint=to_polygon(shape))]))
    return proposals
