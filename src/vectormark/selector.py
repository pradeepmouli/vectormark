"""Geometry candidate generation + scored selection (slice 4a). Replaces
_fit_region's first-non-None cascade with collect-all-then-score: the generator
emits exactly the cascade's fitters (so the selector can only re-prioritise among
known-good geometries, never invent a new output type), in cascade-priority order
(so candidates[0] == the old cascade pick)."""

from __future__ import annotations

import numpy as np

from .candidate import Candidate, FlatFill
from .contour import region_contours
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .refine import (
    half_ellipse_cap_fit, rounded_trapezoid_fit, symmetric_fit, symmetric_polygon_fit,
)
from .score import rank_candidates
from .types import Axis, Region


def _snap_to_axis(shape: Shape, axis: Axis) -> Shape:
    """Force x-centre of a straddling primitive onto the axis for exact symmetry."""
    if shape.kind in ("circle", "ellipse"):
        shape.params["cx"] = axis.x
    elif shape.kind == "rect":
        shape.params["x"] = axis.x - shape.params["w"] / 2
    return shape


def generate_geometry_candidates(
    region: Region, opt, axis: Axis | None, corner_radius: float,
) -> list[Shape]:
    """All geometry fits the cascade could produce for this region, in cascade
    priority order (candidates[0] == the old _fit_region pick), non-None only.

    For a straddler (axis set) the non-symmetric fallbacks (recognize_polygon,
    fit_path) are added ONLY when no symmetric candidate exists — so the scorer
    can never pick a cheaper non-symmetric geometry over a valid symmetric one and
    silently break exact symmetry (matches the cascade's fall-through)."""
    contours = [c for c in region_contours(region.mask) if len(c) >= 3]
    if not contours:
        return []

    if len(contours) > 1:                       # holed / counter
        cands: list[Shape] = []
        if axis is not None:
            halves = [
                symmetric_fit(c, axis.x, corner_radius=corner_radius,
                              epsilon=opt.epsilon, max_error=opt.max_error)
                for c in contours
            ]
            if all(s is not None for s in halves):
                d = " ".join(s.params["d"] for s in halves)
                cands.append(Shape("path", {"d": d, "fill_rule": "evenodd"}))
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error).params["d"]
            for c in contours
        )
        cands.append(Shape("path", {"d": d, "fill_rule": "evenodd"}))
        return cands

    contour = contours[0]
    cands = []

    prim = recognize_primitive(contour, epsilon=opt.epsilon)
    if prim is not None:
        cands.append(_snap_to_axis(prim, axis) if axis is not None else prim)

    sym: list[Shape] = []
    if axis is not None:
        trap = rounded_trapezoid_fit(contour, axis.x, radius=corner_radius, max_error=opt.max_error)
        if trap is not None:
            sym.append(trap)
        poly = symmetric_polygon_fit(contour, axis.x, epsilon=opt.epsilon)
        if poly is not None:
            sym.append(poly)
        cap = half_ellipse_cap_fit(contour, axis.x, corner_radius=corner_radius, max_error=opt.max_error)
        if cap is not None:
            sym.append(cap)
        s = symmetric_fit(contour, axis.x, corner_radius=corner_radius,
                          epsilon=opt.epsilon, max_error=opt.max_error)
        if s is not None:
            sym.append(s)
    cands.extend(sym)

    # Non-symmetric fallbacks: only when there is no symmetry to preserve (axis is
    # None) OR no symmetric candidate was produced. Guarantees a non-empty set.
    if axis is None or not sym:
        gpoly = recognize_polygon(contour, epsilon=opt.epsilon)
        if gpoly is not None:
            cands.append(gpoly)
        cands.append(fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error))

    return cands
