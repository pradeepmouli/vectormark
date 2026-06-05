"""C3/C4: sub-pixel contour extraction, RDP simplification, corner detection."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours


def _polygon_area(c: np.ndarray) -> float:
    """Absolute shoelace area of an (N, 2) closed polygon."""
    x, y = c[:, 0], c[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def region_contours(mask: np.ndarray) -> list[np.ndarray]:
    """All sub-pixel contours of `mask` (outer boundary + any holes), each an
    (N, 2) (x, y) array, sorted by enclosed area descending (outer first)."""
    padded = np.pad(mask.astype(float), 1)
    contours = find_contours(padded, 0.5)               # (row, col) == (y, x)
    out = [np.column_stack([c[:, 1] - 1, c[:, 0] - 1]) for c in contours]  # -> (x,y), unpad
    out.sort(key=_polygon_area, reverse=True)
    return out


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """Longest sub-pixel contour of `mask`, as an (N, 2) array of (x, y) points."""
    padded = np.pad(mask.astype(float), 1)
    contours = find_contours(padded, 0.5)
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)                 # rows of (row, col) == (y, x)
    return np.column_stack([longest[:, 1] - 1, longest[:, 0] - 1])  # -> (x, y), unpad


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
