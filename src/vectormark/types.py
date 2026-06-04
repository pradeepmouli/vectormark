"""Plain data types passed between pipeline stages (IO-free, port-friendly).

`Shape` (a fitted SVG element) lives in `fit.py` next to the code that produces
it; this module holds only the inputs to fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    """One connected, single-colour area of the quantised image."""
    label: int
    mask: np.ndarray          # bool (H, W)
    color_hex: str

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Axis:
    """Vertical axis of bilateral symmetry at image x == self.x."""
    x: float

    def reflect_x(self, x: float) -> float:
        return 2.0 * self.x - x
