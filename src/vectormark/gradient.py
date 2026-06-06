# SPDX-License-Identifier: MIT
"""Gradient detection: recover a gradient's footprint from quantized bands, fit a
linear/radial colour model against the original image, and emit one gradient-filled
shape when it re-renders faithfully (see
docs/superpowers/specs/2026-06-06-gradient-handling-design.md)."""

from __future__ import annotations

import numpy as np

from .color import srgb_to_oklab
from .occlusion import region_adjacency
from .types import Region

_MIN_BANDS = 3
_RAMP_TOL = 0.06          # max OKLab distance of a band colour from the fitted ramp line
_GATE_DELTA_E = 0.05      # mean OKLab ΔE bar (consumed by fit_gradient in a later task)


def _hex_to_oklab(hex_colors: list[str]) -> np.ndarray:
    rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_colors],
        float,
    )
    return srgb_to_oklab(rgb / 255.0)


def _is_ramp(colors_oklab: np.ndarray) -> bool:
    """True if >=3 colours and they lie on a single line in OKLab within _RAMP_TOL."""
    if len(colors_oklab) < _MIN_BANDS:
        return False
    mean = colors_oklab.mean(axis=0)
    centred = colors_oklab - mean
    if np.abs(centred).max() < 1e-6:
        return False                                   # all equal -> flat, not a ramp
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    line = vt[0]                                        # principal colour direction
    proj = np.outer(centred @ line, line)
    resid = np.linalg.norm(centred - proj, axis=1).max()
    return resid <= _RAMP_TOL


def _ramp_groups(regions: list[Region]) -> list[list[Region]]:
    """Connected groups of >=3 adjacent regions whose flat colours form an OKLab ramp."""
    # Grows each full connected component, then tests if the whole thing is a ramp.
    # segment() excludes the background plate, so a gradient mark's bands neighbor only
    # each other -> this works. A gradient bordering another real shape is lumped in and
    # rejected (falls back to flat bands); edge-consistent growth would refine that later.
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    seen: set[int] = set()
    groups: list[list[Region]] = []
    for r in regions:
        if r.label in seen:
            continue
        # grow the connected component
        comp: list[int] = []
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in seen:
                continue
            seen.add(lab)
            comp.append(lab)
            stack.extend(adj[lab] - seen)
        members = [by_label[l] for l in comp]
        if len(members) < _MIN_BANDS:
            continue
        if _is_ramp(_hex_to_oklab([m.color_hex for m in members])):
            groups.append(members)
    return groups
