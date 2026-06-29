"""C1/C2: vertical bilateral symmetry detection + region classification."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy import ndimage as ndi

from .types import Axis, Region

Axis2D = namedtuple("Axis2D", "theta cx cy")


def _perimeter(mask: np.ndarray) -> int:
    """Boundary-pixel count: the one-pixel ring of `mask` minus its erosion."""
    if not mask.any():
        return 0
    return int((mask & ~ndi.binary_erosion(mask)).sum())


def reflection_off_count(fg_xy, axis: Axis2D, dist: np.ndarray, *, tol_px: float = 1.5) -> int:
    """Count foreground points whose reflection across `axis` lands farther than
    `tol_px` outside the shape (background distance transform `dist`). Resampling
    free: reflects coordinates, looks up the distance transform — no raster rotate."""
    xs, ys = fg_xy
    dx, dy = np.cos(axis.theta), np.sin(axis.theta)
    vx, vy = xs - axis.cx, ys - axis.cy
    t = vx * dx + vy * dy
    rx = axis.cx + (2.0 * t * dx - vx)
    ry = axis.cy + (2.0 * t * dy - vy)
    h, w = dist.shape
    ri = np.clip(np.rint(ry).astype(int), 0, h - 1)
    ci = np.clip(np.rint(rx).astype(int), 0, w - 1)
    return int((dist[ri, ci] > tol_px).sum())


K_BAND = 1.5   # absolute boundary-band factor: off-area <= K_BAND * perimeter (~1.5px band)


def _fg_xy(mask):
    ys, xs = np.nonzero(mask)
    return xs, ys


def region_is_self_symmetric(region, axis: Axis2D) -> bool:
    mask = region.mask
    if not mask.any():
        return False
    dist = ndi.distance_transform_edt(~mask)
    off = reflection_off_count(_fg_xy(mask), axis, dist)
    return off <= K_BAND * _perimeter(mask)


def regions_mirror_pair(a, b, axis: Axis2D) -> bool:
    aa, ab = int(a.mask.sum()), int(b.mask.sum())
    if aa == 0 or ab == 0 or min(aa, ab) / max(aa, ab) < 0.9:
        return False
    dist_b = ndi.distance_transform_edt(~b.mask)
    off = reflection_off_count(_fg_xy(a.mask), axis, dist_b)
    return off <= K_BAND * _perimeter(a.mask)


# One tolerance for "is this bilaterally symmetric?", shared by every acceptance site
# so they cannot disagree. It is a reflection *mismatch* fraction: a shape counts as
# symmetric about an axis when at most SYM_TOL of it fails to overlap its reflection
# (equivalently, reflection IoU >= 1 - SYM_TOL). Using a looser bar for the straddle
# gate than for axis detection let near-symmetric *pointed* regions be force-mirrored
# about an axis their apex misses, splitting the apex into a fork.
SYM_TOL = 0.10

# False straddlers (icloud ~0.904, telegram ~0.917) sit well below this threshold;
# genuine self-symmetric marks (gdrive ~0.995, appstore ~0.966) sit at or above it.
# Half-mirroring a SINGLE region erases its real asymmetry, so the straddle gate needs
# a stricter bar than the pair gate (which matches two genuine mirror twins).
STRADDLE_MIN_IOU = 0.96


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
    straddle_iou: float = STRADDLE_MIN_IOU,
    diagnostics: list | None = None,
) -> tuple[list[Region], list[tuple[Region, Region]], list[Region]]:
    """Split regions into self-symmetric straddlers, mirror pairs, and lone
    asymmetric leftovers.

    The three categories must be fit differently, so they are kept distinct: a
    straddler is fit half-and-mirrored about the axis, a pair is fit once and
    `<use>`-mirrored, and a loner is fit as-is with no symmetry (forcing the
    half-outline mirror onto a genuinely asymmetric region would distort it).

    The straddle gate uses a STRICTER bar (STRADDLE_MIN_IOU = 0.96) than the pair
    gate (1 - SYM_TOL = 0.90). Half-mirroring a single region erases its real
    asymmetry; false straddlers (icloud ~0.90, telegram ~0.917) fall below 0.96.
    The pair gate is looser because misclassifying a pair just shows the wrong
    region mirrored, which is less destructive than erasing asymmetry. Below either
    bar, regions fall through to loners (fit as-is). (Corpus check: genuine
    mirror-twin pairs score IoU >= 0.96; false pairs sit at <= 0.74.)"""
    straddlers: list[Region] = []
    pairs: list[tuple[Region, Region]] = []
    loners: list[Region] = []
    used: set[int] = set()
    for r in regions:
        if r.label in used:
            continue
        self_refl = _reflect_cols(r.mask, axis.x)
        self_iou = _iou(r.mask, self_refl)
        if self_iou >= straddle_iou:
            straddlers.append(r)
            used.add(r.label)
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "straddler"))
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
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "pair"))
                partner_refl = _reflect_cols(partner.mask, axis.x)
                diagnostics.append((partner.label, round(float(_iou(partner.mask, partner_refl)), 4), "pair"))
        else:
            loners.append(r)  # asymmetric and unpaired: fit as-is, no forced mirror
            used.add(r.label)
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "loner"))
    return straddlers, pairs, loners
