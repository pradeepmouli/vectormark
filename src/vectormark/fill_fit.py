# SPDX-License-Identifier: MIT
"""Per-shape fill decision: given a shape's silhouette mask and the source pixels,
return the best Fill (flat or parametric gradient). Geometry is never touched here."""

from __future__ import annotations

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill
from .gradient import _GATE_DELTA_E, _best_parametric


def fit_fill(mask: np.ndarray, rgb: np.ndarray, *, flat_hex: str,
             max_gradient_de: float = _GATE_DELTA_E) -> Fill:
    """Decide a shape's fill from the source pixels under `mask`.

    Returns FlatFill(flat_hex) unless a searched parametric gradient (linear or radial)
    both exists and re-renders within `max_gradient_de` mean OKLab ΔE; in that case the
    corresponding gradient fill is returned. The silhouette is the caller's; this only
    chooses how to paint inside it."""
    best = _best_parametric(mask, rgb)
    if best is None:
        return FlatFill(flat_hex)
    model, mean_de, _median_de = best
    if mean_de > max_gradient_de:
        return FlatFill(flat_hex)
    if model["kind"] == "linear":
        return LinearGradientFill(model["geometry"], model["stops"])
    return RadialGradientFill(model["geometry"], model["stops"])
