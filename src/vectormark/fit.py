"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import CircleModel, EllipseModel

from ._fitcurve import fit_cubic_beziers
from .contour import corner_indices, rdp


MAX_PATH_SEGMENTS = 12   # a "shape" is simple; a path needing more drawing commands than
                         # this is fraying (tracing AA noise), not a shape -> disqualified.
MAX_POLY_VERTICES = 10   # a path/polygon "shape" has few corners; more is a traced jagged edge.
PATH_COARSEN_STEPS = 10    # coarsening iterations; 1.6**10 ≈ 110× — enough to collapse any
                           # contour toward a triangle (3 cmds), so any budget >=3 is reachable.
PATH_COARSEN_FACTOR = 1.6  # tolerance growth per step (matches _loose_bounded_polygon).
ROBUST_RESIDUAL_TOL = 1.0   # acceptance multiplier on epsilon for the robust (RMS) residual.


@dataclass
class Shape:
    kind: str                 # "circle" | "ellipse" | "rect" | "polygon" | "path"
    params: dict


def _max_residual(model, pts: np.ndarray) -> float:
    return float(np.abs(model.residuals(pts)).max())


def _robust_residual(model, pts: np.ndarray) -> float:
    """RMS of |perpendicular residuals| — tolerant of a noisy boundary minority,
    unlike the MAX. The least-squares model already fits the bulk; this just accepts
    on the bulk error instead of the worst outlier."""
    r = np.asarray(model.residuals(pts), float)
    return float(np.sqrt(np.mean(r * r)))


def recognize_primitive(contour: np.ndarray, *, epsilon: float) -> Shape | None:
    """Return a native-primitive Shape if `contour` matches one within ε, else None."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 8:
        return None
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 1:
        return None

    # circle
    cm = CircleModel.from_estimate(pts)
    if cm and _robust_residual(cm, pts) <= epsilon * ROBUST_RESIDUAL_TOL:
        xc, yc = cm.center
        return Shape("circle", {"cx": xc, "cy": yc, "r": cm.radius})

    # ellipse (axis-aligned check: snap small thetas to 0 for symmetric output)
    em = EllipseModel.from_estimate(pts)
    if em:
        xc, yc = em.center
        a, b = em.axis_lengths
        theta = em.theta
        if _robust_residual(em, pts) <= epsilon * ROBUST_RESIDUAL_TOL and (abs(theta) < 0.08 or abs(abs(theta) - np.pi) < 0.08):
            return Shape("ellipse", {"cx": xc, "cy": yc, "rx": a, "ry": b})

    # axis-aligned rectangle: bbox fill ratio near 1, rotated-rect ~ axis-aligned,
    # AND genuinely four sharp corners. The corner check is what stops a *rounded
    # trapezoid* (tapered sides + filleted corners) from being flattened to a
    # rect — those simplify to >4 vertices and fall through to the symmetry-locked
    # path fit, which keeps their true shape.
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    rot = poly.minimum_rotated_rectangle
    rx, ry = rot.exterior.xy
    edge_angles = np.arctan2(np.diff(ry), np.diff(rx))
    axis_aligned = np.all(np.minimum(np.abs(edge_angles % (np.pi / 2)),
                                     np.pi / 2 - np.abs(edge_angles % (np.pi / 2))) < 0.06)
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    four_corners = len(simp) == 4
    if bbox_area > 0 and poly.area / bbox_area > 0.985 and axis_aligned and four_corners:
        return Shape("rect", {"x": minx, "y": miny, "w": maxx - minx, "h": maxy - miny})

    return None


def recognize_polygon(contour: np.ndarray, *, epsilon: float, max_vertices: int = MAX_POLY_VERTICES) -> Shape | None:
    """Emit a <polygon> when the contour simplifies to few straight edges."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 3:
        return None
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    if not (3 <= len(simp) <= max_vertices):
        return None
    # every original point must lie within ε of the simplified polygon edges (RMS gate)
    if _robust_point_to_polyline(pts, np.vstack([simp, simp[0]])) > epsilon * ROBUST_RESIDUAL_TOL:
        return None
    return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})


def _max_point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    worst = 0.0
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    for p in pts:
        d = min(_point_seg_dist(p, s[0], s[1]) for s in segs)
        worst = max(worst, d)
    return worst


def _robust_point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    """RMS of each point's distance to the nearest polyline segment (robust counterpart
    of `_max_point_to_polyline`)."""
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    d = np.array([min(_point_seg_dist(p, s[0], s[1]) for s in segs) for p in pts])
    return float(np.sqrt(np.mean(d * d)))


def _point_seg_dist(p, a, b) -> float:
    ab = b - a
    t = 0.0 if np.dot(ab, ab) == 0 else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.hypot(*(p - (a + t * ab))))


def _segment_is_straight(seg: np.ndarray, epsilon: float) -> bool:
    if len(seg) <= 2:
        return True
    return _max_point_to_polyline(seg, np.vstack([seg[0], seg[-1]])) <= epsilon


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _build_corner_path(ring: np.ndarray, epsilon: float, max_error: float) -> tuple[str, int, int]:
    """Build the corner-split path once at the given tolerances.  Returns (d, n_cuts, n_segs)
    so the caller can decide whether the result fits its budget."""
    simp = rdp(ring, epsilon)
    corners = corner_indices(np.vstack([simp, simp[0]]), angle_threshold_deg=40)
    corner_pts = simp[corners] if corners else simp[[0]]
    cut_idx = sorted({int(np.argmin(np.hypot(*(ring - cp).T))) for cp in corner_pts})
    if len(cut_idx) < 2:
        cut_idx = [0, len(ring) // 2]
    d = f"M{_fmt(ring[cut_idx[0]][0])} {_fmt(ring[cut_idx[0]][1])} "
    segs = 0
    for k in range(len(cut_idx)):
        i0 = cut_idx[k]
        i1 = cut_idx[(k + 1) % len(cut_idx)]
        seg = ring[i0:i1 + 1] if i1 > i0 else np.vstack([ring[i0:], ring[: i1 + 1]])
        if len(seg) < 2:
            continue
        if _segment_is_straight(seg, epsilon):
            d += f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
            segs += 1
        else:
            for b in fit_cubic_beziers(seg, max_error):
                d += (f"C{_fmt(b[1][0])} {_fmt(b[1][1])} "
                      f"{_fmt(b[2][0])} {_fmt(b[2][1])} "
                      f"{_fmt(b[3][0])} {_fmt(b[3][1])} ")
                segs += 1
    d += "Z"
    return d, len(cut_idx), segs


def fit_path(contour: np.ndarray, *, epsilon: float, max_error: float,
             max_segments: int = MAX_PATH_SEGMENTS,
             max_vertices: int = MAX_POLY_VERTICES) -> Shape:
    """Corner-split the contour into lines + quadratic Béziers. The segment/vertex limits are
    a complexity BUDGET, not a veto: if a fit exceeds them, the tolerances are coarsened and
    the path refit until it fits — a bounded path is always returned (never None). Coarsening
    also absorbs residual boundary staircase."""
    pts = np.asarray(contour, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    ring = pts[:-1] if closed else pts
    eps, merr = float(epsilon), float(max_error)
    d, ncut, nseg = _build_corner_path(ring, eps, merr)
    for _ in range(PATH_COARSEN_STEPS):
        if ncut <= max_vertices and nseg <= max_segments:
            break
        eps *= PATH_COARSEN_FACTOR
        merr *= PATH_COARSEN_FACTOR
        d, ncut, nseg = _build_corner_path(ring, eps, merr)
    return Shape("path", {"d": d})
