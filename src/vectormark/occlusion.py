# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
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
