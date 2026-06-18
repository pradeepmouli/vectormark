"""C1/C2: vertical bilateral symmetry detection + region classification."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .types import Axis, Region

# One tolerance for "is this bilaterally symmetric?", shared by every acceptance site
# so they cannot disagree. It is a reflection *mismatch* fraction: a shape counts as
# symmetric about an axis when at most SYM_TOL of it fails to overlap its reflection
# (equivalently, reflection IoU >= 1 - SYM_TOL). Using a looser bar for the straddle
# gate than for axis detection let near-symmetric *pointed* regions be force-mirrored
# about an axis their apex misses, splitting the apex into a fork.
SYM_TOL = 0.10


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


def detect_axis(silhouette: np.ndarray, *, tol: float = SYM_TOL) -> Axis | None:
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


def _axis_mismatch(
    fg_xy: tuple[np.ndarray, np.ndarray], cx: float, cy: float, theta: float,
    dist: np.ndarray, *, tol_px: float = 1.5,
) -> float:
    """Fraction of foreground points whose reflection across the line through
    (cx, cy) at angle `theta` lands off the shape (farther than `tol_px`).

    Resampling-free: it reflects point *coordinates* exactly and looks each up in
    the precomputed distance transform of the background, so it does not inherit the
    staircasing a whole-raster rotation would. Reflection preserves area, so
    "reflected ⊆ shape (within tol)" already implies bilateral symmetry about the
    line — no need to also test the reverse direction.
    """
    xs, ys = fg_xy
    dx, dy = np.cos(theta), np.sin(theta)
    vx, vy = xs - cx, ys - cy
    t = vx * dx + vy * dy                       # projection onto the axis direction
    rx = cx + (2.0 * t * dx - vx)               # reflect: keep along-axis, flip ⟂
    ry = cy + (2.0 * t * dy - vy)
    h, w = dist.shape
    ri = np.clip(np.rint(ry).astype(int), 0, h - 1)
    ci = np.clip(np.rint(rx).astype(int), 0, w - 1)
    return float((dist[ri, ci] > tol_px).mean())


def detect_symmetry_rotation(silhouette: np.ndarray, *, tol: float = SYM_TOL) -> float | None:
    """Degrees of rotation that bring a *tilted* mirror axis to vertical, or None.

    For an off-axis bilaterally-symmetric mark the mirror line is one of the two PCA
    principal axes — but which one is not known a priori (the long axis is the
    mirror line for a beet, the short axis for a fat lens). Score *each* principal
    axis with a resampling-free reflection test (refining the angle locally), keep
    the better, and accept only if its off-shape fraction is within `tol`.

    The returned angle `rho` is applied as `ndi.rotate(arr, -rho)` to rectify and
    inverted in SVG as `rotate(-rho)` to wrap the emitted body back into place.
    """
    ys, xs = np.nonzero(silhouette)
    if xs.size < 3:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    pts = np.column_stack([xs - cx, ys - cy]).astype(float)
    _evals, evecs = np.linalg.eigh(pts.T @ pts / len(pts))
    dist = ndi.distance_transform_edt(~silhouette)
    fg_xy = (xs.astype(float), ys.astype(float))
    best: tuple[float, float] | None = None
    for i in range(2):
        base = float(np.arctan2(evecs[1, i], evecs[0, i]))
        # local refinement: PCA is coarse on noisy boundaries
        for dth in np.radians(np.arange(-5.0, 5.0 + 0.5, 0.5)):
            theta = base + dth
            mm = _axis_mismatch(fg_xy, cx, cy, theta, dist)
            if best is None or mm < best[0]:
                best = (mm, theta)
    if best is None or best[0] > tol:
        return None
    # `best[1]` is the mirror-axis angle φ; rotating the image by ndi.rotate(-rho)
    # with rho = -(φ + 90) brings that axis vertical (the 90° accounts for ndi's
    # CCW-in-array sign convention vs the atan2 line angle). The ±180° freedom in φ
    # only flips the rectified mark top-for-bottom — still vertically symmetric.
    return float(-(np.degrees(best[1]) + 90.0))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def classify_regions(
    regions: list[Region], axis: Axis, *, pair_iou: float = 1.0 - SYM_TOL,
    straddle_iou: float = 1.0 - SYM_TOL,
) -> tuple[list[Region], list[tuple[Region, Region]], list[Region]]:
    """Split regions into self-symmetric straddlers, mirror pairs, and lone
    asymmetric leftovers.

    The three categories must be fit differently, so they are kept distinct: a
    straddler is fit half-and-mirrored about the axis, a pair is fit once and
    `<use>`-mirrored, and a loner is fit as-is with no symmetry (forcing the
    half-outline mirror onto a genuinely asymmetric region would distort it).

    Both the straddle gate and the pair gate use the SAME symmetry bar `detect_axis`
    uses (reflection IoU >= 1 - SYM_TOL). Loosening either admits coincidental
    matches: a near-symmetric pointed region forks under half-mirror, and two
    merely-similar regions (e.g. a "B" and an "R") get `<use>`-mirrored — substituting
    one with the mirror of the other. Below the bar, both fall through to loners
    (fit as-is) instead. (Corpus check: genuine mirror-twin pairs score IoU >= 0.96;
    false pairs sit at <= 0.74, so the bar cleanly separates them.)"""
    straddlers: list[Region] = []
    pairs: list[tuple[Region, Region]] = []
    loners: list[Region] = []
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
            loners.append(r)  # asymmetric and unpaired: fit as-is, no forced mirror
            used.add(r.label)
    return straddlers, pairs, loners
