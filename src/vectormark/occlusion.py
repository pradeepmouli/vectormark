# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .types import Region


@dataclass
class ScenePrimitive:
    """A completed shape that may be partially occluded by higher-z primitives."""
    kind: str                 # "circle" | "ellipse"
    params: dict
    color_hex: str
    z: int


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
