"""Symmetry-locked fitting of axis-straddling regions.

A straddling region (the dome, the tip) is fit by its RIGHT-half outline only;
the closed path is then `half + mirror(half)`, so the result is *exactly*
symmetric about the axis — unlike `fit_path` on the raw contour, which inherits
the raster's left/right wobble. Straight runs become `L`, curved runs become
cubic Béziers (Schneider), corners preserved.
"""

from __future__ import annotations

import numpy as np

from ._fitcurve import fit_cubic_beziers
from .contour import rdp
from .fit import Shape, _fmt, _segment_is_straight


def _longest_true_run_circular(mask: np.ndarray) -> tuple[int, int]:
    """(start_index, length) of the longest run of True in a circular bool array."""
    n = len(mask)
    if mask.all():
        return 0, n
    if not mask.any():
        return 0, 0
    off = int(np.argmin(mask))           # rotate to start at a False
    r = np.roll(mask, -off)
    best_start, best_len, i = 0, 0, 0
    while i < n:
        if r[i]:
            j = i
            while j < n and r[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    return (best_start + off) % n, best_len


def _right_half(contour: np.ndarray, axis_x: float) -> np.ndarray | None:
    """The contour arc on the x >= axis side, ordered top (min y) -> bottom."""
    n = len(contour)
    mask = contour[:, 0] >= axis_x
    start, length = _longest_true_run_circular(mask)
    if length < 3:
        return None
    half = contour[[(start + k) % n for k in range(length)]].astype(float)
    if half[0, 1] > half[-1, 1]:          # ensure top -> bottom
        half = half[::-1].copy()
    return half


def _open_corners(poly: np.ndarray, *, angle_threshold_deg: float = 40.0) -> list[int]:
    """Indices of interior vertices of an OPEN polyline whose turn exceeds the threshold."""
    thresh = np.radians(angle_threshold_deg)
    out: list[int] = []
    for i in range(1, len(poly) - 1):
        v1, v2 = poly[i] - poly[i - 1], poly[i + 1] - poly[i]
        if np.hypot(*v1) == 0 or np.hypot(*v2) == 0:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(np.arctan2(cross, np.dot(v1, v2))) >= thresh:
            out.append(i)
    return out


def _fit_open_segments(pts: np.ndarray, epsilon: float, max_error: float):
    """Fit an open polyline to a start point + list of ('L', p) | ('C', c1, c2, p3)."""
    simp = rdp(pts, epsilon)
    corner_pts = simp[_open_corners(simp)] if len(simp) > 2 else np.empty((0, 2))
    cuts = sorted({0, len(pts) - 1} | {
        int(np.argmin(np.hypot(*(pts - cp).T))) for cp in corner_pts
    })
    segs: list[tuple] = []
    for k in range(len(cuts) - 1):
        sub = pts[cuts[k]:cuts[k + 1] + 1]
        if len(sub) < 2:
            continue
        if _segment_is_straight(sub, epsilon):
            segs.append(("L", sub[-1].copy()))
        else:
            for b in fit_cubic_beziers(sub, max_error):
                segs.append(("C", b[1].copy(), b[2].copy(), b[3].copy()))
    return pts[0].copy(), segs


def _emit_symmetric(start: np.ndarray, segs: list[tuple], axis_x: float) -> str:
    """Closed, exactly-symmetric path: forward half, then mirror-reverse the half."""
    f = _fmt

    def mir(p):
        return (2 * axis_x - p[0], p[1])

    starts = [start]
    for s in segs:
        starts.append(s[1] if s[0] == "L" else s[3])

    d = f"M{f(start[0])} {f(start[1])} "
    for s in segs:
        if s[0] == "L":
            d += f"L{f(s[1][0])} {f(s[1][1])} "
        else:
            d += (f"C{f(s[1][0])} {f(s[1][1])} {f(s[2][0])} {f(s[2][1])} "
                  f"{f(s[3][0])} {f(s[3][1])} ")
    for i in range(len(segs) - 1, -1, -1):
        s, p_prev = segs[i], starts[i]
        if s[0] == "L":
            m = mir(p_prev)
            d += f"L{f(m[0])} {f(m[1])} "
        else:
            c2, c1, p0 = mir(s[2]), mir(s[1]), mir(p_prev)
            d += f"C{f(c2[0])} {f(c2[1])} {f(c1[0])} {f(c1[1])} {f(p0[0])} {f(p0[1])} "
    return d + "Z"


def symmetric_fit(contour: np.ndarray, axis_x: float, *, epsilon: float, max_error: float) -> Shape | None:
    """Fit a straddling region's half-outline and mirror it → exactly-symmetric path."""
    half = _right_half(contour, axis_x)
    if half is None or len(half) < 3:
        return None
    start, segs = _fit_open_segments(half, epsilon, max_error)
    if not segs:
        return None
    start[0] = axis_x                      # pin apex to the axis
    last = segs[-1]
    (last[1] if last[0] == "L" else last[3])[0] = axis_x   # pin bottom to the axis
    return Shape("path", {"d": _emit_symmetric(start, segs, axis_x)})
