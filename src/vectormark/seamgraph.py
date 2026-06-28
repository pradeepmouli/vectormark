"""Seam-graph: shared-edge planar map for gap-free adjacent-region fitting.

Phase A: pure module — no pipeline integration.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Contour side-label classification
# ---------------------------------------------------------------------------

def _bilinear_sample(L: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly sample (H,W,K) array at sub-pixel point (x, y)."""
    H, W, K = L.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    dx = x - np.floor(x); dy = y - np.floor(y)
    return ((1 - dy) * (1 - dx) * L[y0c, x0c]
            + (1 - dy) * dx * L[y0c, x1c]
            + dy * (1 - dx) * L[y1c, x0c]
            + dy * dx * L[y1c, x1c])


def classify_contour(
    contour: np.ndarray,
    L: np.ndarray,
    region_idx: int,
    *,
    bg_idx: int,
) -> np.ndarray:
    """For each point of contour (N,2) (x,y), return the integer label across
    the boundary — the argmax of L at that sub-pixel point among labels ≠ region_idx.
    Value-ordered argmax ties (prefer lower index)."""
    K = L.shape[2]
    N = len(contour)
    result = np.empty(N, dtype=np.intp)
    bias = np.arange(K) * 1e-12          # value-ordered tie-break: prefer lower idx
    for i in range(N):
        x, y = float(contour[i, 0]), float(contour[i, 1])
        lv = _bilinear_sample(L, x, y) + bias
        lv[region_idx] = -np.inf
        result[i] = int(np.argmax(lv))
    return result
