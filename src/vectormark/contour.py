"""C3/C4: sub-pixel contour extraction, RDP simplification, corner detection."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours

_CORNER_RDP_EPS = 2.0        # contour->polygon simplification tolerance (px): 2.0 avoids arc fragmentation
_CORNER_STRAIGHT_TOL = 2.5   # max TLS perpendicular residual (px) for an edge to count as a straight side
_CORNER_MAX_FILLET_FRAC = 0.30  # max inset as a fraction of the shorter adjacent edge (a fillet, not a cap)
_CORNER_MIN_FILLET = 2.5     # min angle-corrected radius (px) to treat a corner as rounded; below -> sharp (0)
_CORNER_DEANTIALIAS_PAD = 2.0  # added to a DETECTED fillet radius only (never to a sharp 0)


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


def _slice_loop(contour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Points of a closed contour from index i to j inclusive, handling wraparound."""
    return contour[i:j + 1] if i <= j else np.vstack([contour[i:], contour[:j + 1]])


def _edge_line(edge: np.ndarray):
    """Total-least-squares line over the middle 60% of an edge (excludes the corner
    transitions at both ends). Returns (centroid, unit_direction, max_perp_residual) or
    None if the edge is too short."""
    k = len(edge)
    if k < 5:
        return None
    mid = edge[k // 5: k - k // 5]
    if len(mid) < 3:
        return None
    c = mid.mean(axis=0)
    _, _, vt = np.linalg.svd(mid - c, full_matrices=False)
    d = vt[0]
    perp = np.abs((mid - c) @ np.array([-d[1], d[0]]))
    return c, d, float(perp.max())


def _line_intersection(c1, d1, c2, d2):
    """Intersection of lines (c1 + t*d1) and (c2 + s*d2), or None if near-parallel."""
    a = np.column_stack([d1, -d2])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, c2 - c1)
    return c1 + t[0] * d1


def _corner_radius_at(before: np.ndarray, after: np.ndarray, vertex: np.ndarray):
    """Angle-corrected fillet radius at one polygon corner, or None if it is not a clean
    corner. `before`/`after` are the contour points on the two edges meeting at `vertex`.

    Fit a straight line to each edge; intersect them at the sharp-corner point P; measure
    the inset (nearest contour point near the vertex to P); convert that inset to a true
    radius using the corner angle (for a fillet radius r in a corner of interior angle
    theta, the contour's nearest approach to P is r*(1/sin(theta/2) - 1))."""
    l1, l2 = _edge_line(before), _edge_line(after)
    if l1 is None or l2 is None:
        return None
    (c1, d1, r1), (c2, d2, r2) = l1, l2
    if r1 > _CORNER_STRAIGHT_TOL or r2 > _CORNER_STRAIGHT_TOL:
        return None                                  # an edge isn't a straight side
    p = _line_intersection(c1, d1, c2, d2)
    if p is None:
        return None
    near = np.vstack([before[-3:], after[:3], vertex[None]])
    inset = float(np.min(np.hypot(near[:, 0] - p[0], near[:, 1] - p[1])))
    edge_len = min(float(np.hypot(*(before[0] - before[-1]))),
                   float(np.hypot(*(after[0] - after[-1]))))
    if edge_len <= 0 or inset > _CORNER_MAX_FILLET_FRAC * edge_len:
        return None                                  # too deep -> a cap, not a corner fillet
    # interior angle between the two edges (directions point from P into each edge body)
    m1 = before[:max(1, len(before) // 2)].mean(axis=0) - p
    m2 = after[len(after) // 2:].mean(axis=0) - p
    n1, n2 = np.hypot(*m1) or 1.0, np.hypot(*m2) or 1.0
    theta = float(np.arccos(np.clip((m1 / n1) @ (m2 / n2), -1.0, 1.0)))
    if theta <= 1e-2:
        return None
    denom = 1.0 / np.sin(theta / 2.0) - 1.0
    if denom < 0.05:                                 # near-straight join: not a real corner
        return None
    return inset / denom


def region_corner_radius(mask: np.ndarray) -> float:
    """One representative corner-fillet radius (px) for the shape in `mask`, measured from
    its outer contour; 0.0 for a sharp-cornered shape. rdp-approximates the contour to
    find its corners, measures the angle-corrected fillet radius at each (see
    _corner_radius_at), and returns the median — padded by _CORNER_DEANTIALIAS_PAD only
    when a real fillet is detected, so a sharp corner reads exactly 0.0."""
    cs = region_contours(mask)
    if not cs or len(cs[0]) < 12:
        return 0.0
    contour = cs[0]
    verts = rdp(contour, _CORNER_RDP_EPS)
    if len(verts) >= 2 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]                           # drop the duplicated closing point
    n = len(verts)
    if n < 3:
        return 0.0
    idx = [int(np.argmin(np.sum((contour - v) ** 2, axis=1))) for v in verts]
    radii: list[float] = []
    for i in range(n):
        before = _slice_loop(contour, idx[(i - 1) % n], idx[i])
        after = _slice_loop(contour, idx[i], idx[(i + 1) % n])
        r = _corner_radius_at(before, after, contour[idx[i]])
        if r is not None:
            radii.append(r)
    if not radii:
        return 0.0
    radii.sort()
    median_r = radii[len(radii) // 2]
    if median_r < _CORNER_MIN_FILLET:
        return 0.0                                   # sharp -> exactly 0 (no pad)
    return round(median_r + _CORNER_DEANTIALIAS_PAD, 1)
