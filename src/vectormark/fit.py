"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import CircleModel, EllipseModel

from ._fitcurve import fit_cubic_beziers, fit_quadratic_beziers
from .contour import corner_indices, rdp

# Curved runs are fit with quadratic Béziers by default: a parabola cannot
# inflect, so it averages the quantization staircase into smooth convex arcs.
# Cubics (opt-in, see fit_path's `cubic` flag) carry an extra DOF that better
# represents genuinely complex contours, but on a staircased raster that DOF
# just chases the noise (rippled/fragmented edges) — they are a complexity
# tool, not a denoiser. When cubics ARE requested, RDP-denoise each run first
# at sub-pixel tolerance to collapse the staircase before fitting.
PATH_DENOISE_EPS = 0.5


@dataclass
class Shape:
    kind: str                 # "circle" | "ellipse" | "rect" | "polygon" | "path"
    params: dict


def _max_residual(model, pts: np.ndarray) -> float:
    return float(np.abs(model.residuals(pts)).max())


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
    if cm and _max_residual(cm, pts) <= epsilon:
        xc, yc = cm.center
        return Shape("circle", {"cx": xc, "cy": yc, "r": cm.radius})

    # ellipse (axis-aligned check: snap small thetas to 0 for symmetric output)
    em = EllipseModel.from_estimate(pts)
    if em:
        xc, yc = em.center
        a, b = em.axis_lengths
        theta = em.theta
        if _max_residual(em, pts) <= epsilon and (abs(theta) < 0.08 or abs(abs(theta) - np.pi) < 0.08):
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


def recognize_polygon(contour: np.ndarray, *, epsilon: float, max_vertices: int = 8) -> Shape | None:
    """Emit a <polygon> when the contour simplifies to few straight edges."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 3:
        return None
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    if not (3 <= len(simp) <= max_vertices):
        return None
    # every original point must lie within ε of the simplified polygon edges
    if _max_point_to_polyline(pts, np.vstack([simp, simp[0]])) > epsilon:
        return None
    return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})


def _max_point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    worst = 0.0
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    for p in pts:
        d = min(_point_seg_dist(p, s[0], s[1]) for s in segs)
        worst = max(worst, d)
    return worst


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


def _append_quadratic_run(d: str, seg: np.ndarray, max_error: float) -> str:
    for b in fit_quadratic_beziers(seg, max_error):
        d += f"Q{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
    return d


def _append_cubic_run(d: str, seg: np.ndarray, max_error: float) -> str:
    run = rdp(seg, PATH_DENOISE_EPS) if len(seg) > 2 else seg
    for b in fit_cubic_beziers(run, max_error):
        d += (
            f"C{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
            f"{_fmt(b[3][0])} {_fmt(b[3][1])} "
        )
    return d


def _curved_run_d(seg: np.ndarray, max_error: float, *, cubic: bool) -> str:
    quadratic = _append_quadratic_run("", seg, max_error)
    if not cubic:
        return quadratic
    cubic_d = _append_cubic_run("", seg, max_error)
    if len(cubic_d.encode()) < len(quadratic.encode()):
        return cubic_d
    return quadratic


def fit_path(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    forced_corners: np.ndarray | None = None,
) -> Shape:
    """Corner-split the contour; emit lines for straight runs, Béziers otherwise.

    ``cubic=False`` (default) fits each curved run with inflection-free
    quadratics — robust against the quantization staircase. ``cubic=True`` still
    tries quadratics first and only keeps denoised, inflection-guarded cubics
    when they produce shorter path data; see PATH_DENOISE_EPS.
    """
    pts = np.asarray(contour, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    ring = pts[:-1] if closed else pts
    simp = rdp(ring, epsilon)
    corners = corner_indices(np.vstack([simp, simp[0]]), angle_threshold_deg=40)
    # map corner positions in `simp` back to indices in `ring`
    corner_pts = simp[corners] if corners else simp[[0]]
    if forced_corners is not None and len(forced_corners):
        corner_pts = np.vstack([corner_pts, np.asarray(forced_corners, dtype=float)])
    cut_idx = sorted({int(np.argmin(np.hypot(*(ring - cp).T))) for cp in corner_pts})
    if len(cut_idx) < 2:
        cut_idx = [0, len(ring) // 2]

    d = f"M{_fmt(ring[cut_idx[0]][0])} {_fmt(ring[cut_idx[0]][1])} "
    for k in range(len(cut_idx)):
        i0 = cut_idx[k]
        i1 = cut_idx[(k + 1) % len(cut_idx)]
        seg = ring[i0:i1 + 1] if i1 > i0 else np.vstack([ring[i0:], ring[: i1 + 1]])
        if len(seg) < 2:
            continue
        if _segment_is_straight(seg, epsilon):
            d += f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
        else:
            d += _curved_run_d(seg, max_error, cubic=cubic)
    d += "Z"
    return Shape("path", {"d": d})
