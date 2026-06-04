"""C3/C4: sub-pixel contour extraction, RDP simplification, corner detection."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """Longest sub-pixel contour of `mask`, as an (N, 2) array of (x, y) points."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)          # rows of (row, col) == (y, x)
    return np.column_stack([longest[:, 1], longest[:, 0]])  # -> (x, y)


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    line = end - start
    norm = np.hypot(*line)
    if norm == 0:
        d = np.hypot(*(pts - start).T)
    else:
        diff = pts - start
        # 2-D cross product as an explicit scalar (np.cross on 2-D vectors is
        # deprecated in NumPy 2.0)
        d = np.abs(line[0] * diff[:, 1] - line[1] * diff[:, 0]) / norm
    idx = int(d.argmax())
    if d[idx] > epsilon:
        left = rdp(pts[: idx + 1], epsilon)
        right = rdp(pts[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def corner_indices(poly: np.ndarray, *, angle_threshold_deg: float = 40.0) -> list[int]:
    """Indices of `poly` vertices whose turn angle exceeds the threshold.

    `poly` is assumed closed (first point repeated at the end).
    """
    pts = np.asarray(poly, dtype=float)
    n = len(pts) - 1 if np.allclose(pts[0], pts[-1]) else len(pts)
    thresh = np.radians(angle_threshold_deg)
    corners: list[int] = []
    for i in range(n):
        prev, cur, nxt = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n]
        v1, v2 = cur - prev, nxt - cur
        if np.hypot(*v1) == 0 or np.hypot(*v2) == 0:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]   # 2-D scalar cross (NumPy 2.0)
        ang = np.arctan2(cross, np.dot(v1, v2))
        if abs(ang) >= thresh:
            corners.append(i)
    return corners
