# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import CircleModel, EllipseModel
from skimage.morphology import convex_hull_image

from .contour import region_contours
from .types import Region


@dataclass
class ScenePrimitive:
    """A completed shape that may be partially occluded by higher-z primitives."""
    kind: str                 # "circle" | "ellipse"
    params: dict
    color_hex: str
    z: int


def has_bite(mask: np.ndarray, *, max_solidity: float = 0.92) -> bool:
    """True when the region is non-convex enough to be a plausible occluded fragment
    (a crescent), i.e. its solidity (area / convex-hull area) is below the bar."""
    area = int(mask.sum())
    if area == 0:
        return False
    hull = int(convex_hull_image(mask).sum())
    return hull > 0 and (area / hull) < max_solidity


def label_boundary(
    region: Region, others: list[Region], *, reach: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer_contour Nx2 as (x,y), seam_bool N). A contour point is a seam
    if any OTHER region's mask sits within `reach` px of it; else it is own boundary."""
    contours = region_contours(region.mask)
    if not contours:
        return np.empty((0, 2)), np.empty((0,), bool)
    contour = contours[0]
    if not others:
        return contour, np.zeros(len(contour), bool)
    near = np.zeros_like(region.mask)
    for o in others:
        near |= binary_dilation(o.mask, iterations=reach)
    h, w = region.mask.shape
    xs = np.clip(np.rint(contour[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(contour[:, 1]).astype(int), 0, h - 1)
    seam = near[ys, xs]
    return contour, seam


def region_adjacency(regions: list[Region]) -> dict[int, set[int]]:
    """label -> set of labels whose masks touch it (8-connectivity, 1px dilation)."""
    adj: dict[int, set[int]] = {r.label: set() for r in regions}
    dilated = {r.label: binary_dilation(r.mask) for r in regions}
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if (dilated[a.label] & b.mask).any():
                adj[a.label].add(b.label)
                adj[b.label].add(a.label)
    return adj


def _own_arc_span_deg(own_pts: np.ndarray, cx: float, cy: float) -> float:
    """Angular span (deg) of the own points about (cx, cy). Full circle -> ~360."""
    ang = np.sort(np.arctan2(own_pts[:, 1] - cy, own_pts[:, 0] - cx))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(np.degrees(2 * np.pi - gaps.max()))     # span covered = full minus largest gap


def _fit_candidate_pts(own: np.ndarray) -> np.ndarray:
    """Return the convex-hull vertices of `own` when feasible, else `own` itself.
    Using the convex hull isolates the outer perimeter, discarding inner concave arcs
    that arise when seam detection cannot mark the full interior boundary."""
    from scipy.spatial import ConvexHull, QhullError
    if len(own) < 4:
        return own
    try:
        hull = ConvexHull(own)
        return own[hull.vertices]
    except QhullError:
        return own


def complete_primitive(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle (then ellipse) to the OWN-boundary points, completing across the
    seam. Returns {"kind","params"} or None when the own arc can't constrain a fit.

    Fitting is performed on the convex hull of the own points so that inner concave
    arcs (which occur when the occluder's overlap zone is not passed as a neighbor
    region and thus not captured by seam detection) do not bias the fit."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    cm = CircleModel.from_estimate(fit_pts)
    if cm and np.abs(cm.residuals(fit_pts)).max() <= max_residual:
        cx, cy = float(cm.center[0]), float(cm.center[1])
        if _own_arc_span_deg(fit_pts, cx, cy) >= min_arc_deg:
            seam_pts = np.asarray(contour, float)[seam]
            inside = len(seam_pts) == 0 or np.all(
                (seam_pts[:, 0] - cx) ** 2 + (seam_pts[:, 1] - cy) ** 2 <= (cm.radius + max_residual) ** 2
            )
            if inside:
                return {"kind": "circle", "params": {"cx": cx, "cy": cy, "r": float(cm.radius)}}
    em = EllipseModel.from_estimate(fit_pts)
    if em and np.abs(em.residuals(fit_pts)).max() <= max_residual:
        xc, yc = float(em.center[0]), float(em.center[1])
        a, b = (float(v) for v in em.axis_lengths)
        if abs(em.theta) < 0.08 or abs(abs(em.theta) - np.pi) < 0.08:
            if _own_arc_span_deg(fit_pts, xc, yc) >= min_arc_deg:
                return {"kind": "ellipse", "params": {"cx": xc, "cy": yc, "rx": a, "ry": b}}
    return None
