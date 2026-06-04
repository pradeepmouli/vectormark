"""A0 colour optimisation: OKLab conversion, ΔE, palette extraction, quantise."""

from __future__ import annotations

import numpy as np

# sRGB <-> linear
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# OKLab matrices (Björn Ottosson, 2020)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) in [0,1] -> OKLab (..., 3)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    lms = lin @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> sRGB (..., 3) clipped to [0,1]."""
    lms_ = np.asarray(lab, dtype=float) @ _M2_INV.T
    lms = lms_ ** 3
    lin = lms @ _M1_INV.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """OKLab Euclidean distance (OKLab is designed for L2 ΔE)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
