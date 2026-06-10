# src/vectormark/variants.py
"""Whole-mark variant matrix: idealize one raster across an epsilon × max_error
grid, for SVG export, a JSON manifest, and an annotated contact sheet."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .pipeline import IdealizeReport, Options, _flatten_on_white, idealize

DEFAULT_EPSILONS = (0.5, 1.5, 3.0)
DEFAULT_MAX_ERRORS = (0.5, 1.0, 2.5)


@dataclass(frozen=True)
class Variant:
    """One matrix cell: the (epsilon, max_error) used, the emitted SVG, the
    strategy report, and an error string if that cell failed to idealize."""

    epsilon: float
    max_error: float
    svg: str
    report: IdealizeReport
    error: str | None = None


def _as_rgb(image) -> np.ndarray:
    """Load any accepted image input to an (H, W, 3) uint8 array ONCE, so the grid
    does not re-read/re-flatten the source per cell."""
    if isinstance(image, str):
        with Image.open(image) as im:
            return _flatten_on_white(im)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _flatten_on_white(Image.fromarray(arr, "RGBA"))
    return arr


def generate_variants(
    image, *, epsilons=DEFAULT_EPSILONS, max_errors=DEFAULT_MAX_ERRORS,
    base: Options | None = None,
) -> list[Variant]:
    """idealize() the image once per (epsilon, max_error) cell, row-major (epsilon
    outer, max_error inner). `base` supplies the non-geometry knobs held constant
    across the grid (max_colors, no_symmetry, …); epsilon/max_error are overridden
    per cell. A cell that raises is captured as a failed Variant, never aborting
    the grid."""
    arr = _as_rgb(image)
    base = base or Options()
    out: list[Variant] = []
    for eps in epsilons:
        for me in max_errors:
            opt = replace(base, epsilon=eps, max_error=me)
            try:
                svg, report = idealize(arr, options=opt, report=True)
                out.append(Variant(eps, me, svg, report))
            except Exception as exc:                       # one bad cell must not kill the grid
                out.append(Variant(eps, me, "", IdealizeReport.empty(), error=str(exc)))
    return out
