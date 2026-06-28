# SPDX-License-Identifier: MIT
"""Antialiasing-aware soft label field: recover per-region sub-pixel coverage from the
pre-quantization RGB + palette, so contour extraction is smooth and seams are shared.
Pure numpy; no pipeline imports."""

from __future__ import annotations

import numpy as np


def alpha_unmix(rgb: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> np.ndarray:
    """Coverage of color A in a two-color blend V = α·c_a + (1−α)·c_b.
    α = clip((V−c_b)·(c_a−c_b)/|c_a−c_b|², 0, 1). α=1 ⇒ pure c_a. Vectorized over leading
    dims of `rgb` (last axis = channels)."""
    rgb = np.asarray(rgb, float); c_a = np.asarray(c_a, float); c_b = np.asarray(c_b, float)
    d = c_a - c_b
    denom = float(d @ d) or 1.0
    return np.clip(((rgb - c_b) @ d) / denom, 0.0, 1.0)
