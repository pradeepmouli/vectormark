"""B: split the quantised image into connected single-colour regions."""

from __future__ import annotations

import numpy as np
from skimage.measure import label

from .types import Region


def hexstr(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _background_color(q: np.ndarray) -> tuple[int, int, int]:
    """Majority colour on the 1px border = background plate."""
    border = np.concatenate([q[0], q[-1], q[:, 0], q[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return tuple(int(v) for v in colors[counts.argmax()])


def segment(quantized: np.ndarray, *, min_area: int = 16) -> list[Region]:
    """Connected components per palette colour, excluding the background."""
    bg = _background_color(quantized)
    palette = np.unique(quantized.reshape(-1, 3), axis=0)
    regions: list[Region] = []
    next_label = 1
    for color in palette:
        if tuple(int(v) for v in color) == bg:
            continue
        color_mask = np.all(quantized == color, axis=2)
        labels = label(color_mask, connectivity=2)
        for lab_id in range(1, labels.max() + 1):
            comp = labels == lab_id
            if comp.sum() < min_area:
                continue
            regions.append(Region(next_label, comp, hexstr(tuple(int(v) for v in color))))
            next_label += 1
    return regions
