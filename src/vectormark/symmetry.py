"""C1/C2: vertical bilateral symmetry detection + region classification."""

from __future__ import annotations

import numpy as np

from .types import Axis, Region


def _reflect_cols(mask: np.ndarray, axis_x: float) -> np.ndarray:
    """Reflect a boolean mask across the vertical line x = axis_x (nearest col)."""
    h, w = mask.shape
    cols = np.arange(w)
    src = np.rint(2.0 * axis_x - cols).astype(int)
    valid = (src >= 0) & (src < w)
    out = np.zeros_like(mask)
    out[:, valid] = mask[:, src[valid]]
    return out


def _mismatch(mask: np.ndarray, axis_x: float) -> float:
    refl = _reflect_cols(mask, axis_x)
    inter = np.logical_and(mask, refl).sum()
    union = np.logical_or(mask, refl).sum()
    return 1.0 - (inter / union if union else 1.0)


def detect_axis(silhouette: np.ndarray, *, tol: float = 0.10) -> Axis | None:
    """Find the best vertical symmetry axis; None if mismatch exceeds `tol`.

    Searches candidate columns near the foreground centroid at 0.5px resolution.
    """
    ys, xs = np.nonzero(silhouette)
    if xs.size == 0:
        return None
    cx = round(float(xs.mean()))
    candidates = np.arange(cx - 6, cx + 6 + 0.5, 0.5)
    scored = [(float(_mismatch(silhouette, a)), float(a)) for a in candidates]
    best_mismatch, best_x = min(scored)
    return Axis(x=best_x) if best_mismatch <= tol else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def classify_regions(
    regions: list[Region], axis: Axis, *, pair_iou: float = 0.6, straddle_iou: float = 0.5,
) -> tuple[list[Region], list[tuple[Region, Region]]]:
    """Split regions into straddlers (self-symmetric) and mirror pairs."""
    straddlers: list[Region] = []
    pairs: list[tuple[Region, Region]] = []
    used: set[int] = set()
    for r in regions:
        if r.label in used:
            continue
        self_refl = _reflect_cols(r.mask, axis.x)
        if _iou(r.mask, self_refl) >= straddle_iou:
            straddlers.append(r)
            used.add(r.label)
            continue
        # find a partner whose mask matches r's reflection
        partner = None
        for other in regions:
            if other.label in used or other.label == r.label:
                continue
            if other.color_hex == r.color_hex and _iou(self_refl, other.mask) >= pair_iou:
                partner = other
                break
        if partner is not None:
            # canonical = the one whose centroid is on the +x (right) side
            canon, mirror = (r, partner)
            if np.nonzero(r.mask)[1].mean() < axis.x:
                canon, mirror = partner, r
            pairs.append((canon, mirror))
            used.update({r.label, partner.label})
        else:
            straddlers.append(r)  # lone asymmetric region: fit as-is
            used.add(r.label)
    return straddlers, pairs
