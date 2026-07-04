# SPDX-License-Identifier: MIT
"""Antialiasing-aware soft label field: recover per-region sub-pixel coverage from the
pre-quantization RGB + palette, so contour extraction is smooth and seams are shared.
No pipeline imports."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .color import srgb_to_oklab

BAND_REACH = 2   # px each side of a label transition treated as an antialiasing band


def alpha_unmix(rgb: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> np.ndarray:
    """Coverage of color A in a two-color blend V = α·c_a + (1−α)·c_b.
    α = clip((V−c_b)·(c_a−c_b)/|c_a−c_b|², 0, 1). α=1 ⇒ pure c_a. Vectorized over leading
    dims of `rgb` (last axis = channels)."""
    rgb = np.asarray(rgb, float)
    c_a = np.asarray(c_a, float)
    c_b = np.asarray(c_b, float)
    d = c_a - c_b
    denom = np.sum(d * d, axis=-1)
    denom = np.where(denom == 0.0, 1.0, denom)
    numer = np.sum((rgb - c_b) * d, axis=-1)
    return np.clip(numer / denom, 0.0, 1.0)


def _oklab_dist(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H,W,K) OKLab distance from each pixel to each palette color."""
    px = srgb_to_oklab(rgb / 255.0)                        # (H,W,3)
    pal = srgb_to_oklab(np.asarray(palette, float) / 255.0)  # (K,3)
    return np.linalg.norm(px[:, :, None, :] - pal[None, None, :, :], axis=3)


def soft_label_field(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Per-pixel partition-of-unity membership in each palette color (background included
    as a row). Interior one-hot (anchors thin features); two-color band alpha-unmixed;
    >=3-color junction band normalized-inverse-ΔE. Deterministic (value-ordered ties)."""
    rgb = np.asarray(rgb, float)
    palette = np.asarray(palette, float)
    H, W, _ = rgb.shape
    K = len(palette)
    dist = _oklab_dist(rgb, palette)                       # (H,W,K)
    # rank the two nearest labels per pixel; value-ordered ties via a tiny index bias
    order = np.argsort(dist + np.arange(K) * 1e-12, axis=2)  # (H,W,K) ascending
    n0 = order[..., 0]; n1 = order[..., 1]
    d0 = np.take_along_axis(dist, n0[..., None], 2)[..., 0]
    d1 = np.take_along_axis(dist, n1[..., None], 2)[..., 0]

    # spatial band: a pixel is an AA-boundary candidate only if it is within BAND_REACH
    # of a nearest-label (n0) transition; solid interiors are one-hot regardless of
    # palette twin colors in OKLab (replaces color-only d0 < 0.5*d1 heuristic).
    labels = n0                                              # (H,W) hard nearest-palette label
    trans = np.zeros((H, W), bool)
    ud = labels[:-1, :] != labels[1:, :]
    trans[:-1, :] |= ud; trans[1:, :] |= ud
    lr = labels[:, :-1] != labels[:, 1:]
    trans[:, :-1] |= lr; trans[:, 1:] |= lr
    band = ndimage.binary_dilation(trans, iterations=BAND_REACH)  # AA-boundary candidates
    interior = ~band

    L = np.zeros((H, W, K), float)
    np.put_along_axis(L, n0[..., None], np.where(interior, 1.0, 0.0)[..., None], 2)

    # boundary band (not interior): unmix the two locally-dominant colors
    if band.any():
        by, bx = np.where(band)
        ca = palette[n0[by, bx]]
        cb = palette[n1[by, bx]]
        a = alpha_unmix(rgb[by, bx], ca, cb)
        L[by, bx, n0[by, bx]] = a
        L[by, bx, n1[by, bx]] = 1.0 - a

    # junction band: a pixel with >=3 comparably-near labels is ill-posed for unmix;
    # detect (3rd-nearest within 1.3x of nearest) and overwrite with inverse-ΔE membership.
    d2 = np.take_along_axis(dist, order[..., 2][..., None], 2)[..., 0] if K >= 3 else np.full((H, W), np.inf)
    junction = band & (d2 < 1.3 * np.maximum(d0, 1e-9))
    if junction.any():
        inv = 1.0 / (dist[junction] + 1e-6)
        L[junction] = inv / inv.sum(axis=1, keepdims=True)

    return L


def region_coverage(L: np.ndarray, k: int, region_mask: np.ndarray, *, reach: int = 2) -> np.ndarray:
    """Region k's coverage field from the shared global L: cov = (φ+1)/2 with
    φ = L[...,k] − max_{j≠k} L[...,j] (boundary at 0.5), zeroed outside a `reach`-px
    dilation of `region_mask` so only this component's boundary is traced. Derived from
    the SAME L for every region ⇒ φ_B = −φ_A on shared seams (gap-free)."""
    others = np.delete(L, k, axis=2).max(axis=2)
    phi = L[..., k] - others
    cov = (phi + 1.0) / 2.0
    near = ndimage.binary_dilation(region_mask, iterations=reach)
    cov = np.where(near, cov, 0.0)
    return cov
