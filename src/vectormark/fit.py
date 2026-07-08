"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import CircleModel, EllipseModel

from ._fitcurve import cbezier, cubic_inflects, fit_cubic_beziers, fit_quadratic_beziers
from .contour import corner_indices, rdp
from .skia_geometry import SkPath

# Curved runs are fit with quadratic Béziers by default: a parabola cannot
# inflect, so it averages the quantization staircase into smooth convex arcs.
# Cubics (opt-in, see fit_path's `cubic` flag) carry an extra DOF that better
# represents genuinely complex contours, but on a staircased raster that DOF
# just chases the noise (rippled/fragmented edges) — they are a complexity
# tool, not a denoiser. When cubics ARE requested, RDP-denoise each run first
# at sub-pixel tolerance to collapse the staircase before fitting.
PATH_DENOISE_EPS = 0.5
MIN_LINE_LENGTH_FACTOR = 8.0
SIMPLE_CURVE_RESIDUAL_TOL = 0.012


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
    poly = SkPath(shell=list(map(tuple, pts)))
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


def _segment_length(seg: np.ndarray) -> float:
    return float(np.linalg.norm(seg[-1] - seg[0]))


def minimum_line_length(epsilon: float) -> float:
    return max(4.0, epsilon * MIN_LINE_LENGTH_FACTOR)


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _path_command_count(d: str) -> int:
    return sum(1 for char in d if char in "MLQCAZ")


def _closed_ring_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        return pts[:-1]
    return pts


def _path_area_residual(contour: np.ndarray, d: str) -> float:
    ring = _closed_ring_points(np.asarray(contour, dtype=float))
    if len(ring) < 3:
        return float("inf")
    try:
        source = SkPath(shell=list(map(tuple, ring)))
        fitted = SkPath.from_svg_d(d)
        if source.is_empty or fitted.is_empty:
            return float("inf")
        scale = max(float(source.area), float(fitted.area), 1.0)
        return float(source.symmetric_difference(fitted).area / scale)
    except Exception:
        return float("inf")


def _curve_controls_are_straight(start: np.ndarray, controls: list[np.ndarray], end: np.ndarray, epsilon: float) -> bool:
    if float(np.linalg.norm(end - start)) < minimum_line_length(epsilon):
        return False
    return _max_point_to_polyline(np.asarray(controls, dtype=float), np.vstack([start, end])) <= epsilon


def _append_quadratic_run(
    d: str,
    seg: np.ndarray,
    max_error: float,
    *,
    line_epsilon: float,
    collapse_straight: bool = True,
) -> str:
    curves = fit_quadratic_beziers(seg, max_error)
    for b in curves:
        if collapse_straight and _curve_controls_are_straight(b[0], [b[1]], b[2], line_epsilon):
            d += f"L{_fmt(b[2][0])} {_fmt(b[2][1])} "
        else:
            d += f"Q{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
    return d


def _append_cubic_run(d: str, seg: np.ndarray, max_error: float, *, line_epsilon: float) -> str:
    run = rdp(seg, PATH_DENOISE_EPS) if len(seg) > 2 else seg
    current = np.array([float(_fmt(seg[0][0])), float(_fmt(seg[0][1]))], dtype=float)
    for b in fit_cubic_beziers(run, max_error):
        rounded = np.array(
            [
                current,
                [float(_fmt(b[1][0])), float(_fmt(b[1][1]))],
                [float(_fmt(b[2][0])), float(_fmt(b[2][1]))],
                [float(_fmt(b[3][0])), float(_fmt(b[3][1]))],
            ],
            dtype=float,
        )
        if cubic_inflects(rounded):
            samples = np.asarray([cbezier(b, t) for t in np.linspace(0.0, 1.0, 9)], dtype=float)
            d = _append_quadratic_run(d, samples, max_error, line_epsilon=line_epsilon, collapse_straight=False)
            current = np.array([float(_fmt(samples[-1][0])), float(_fmt(samples[-1][1]))], dtype=float)
            continue
        d += (
            f"C{_fmt(rounded[1][0])} {_fmt(rounded[1][1])} {_fmt(rounded[2][0])} {_fmt(rounded[2][1])} "
            f"{_fmt(rounded[3][0])} {_fmt(rounded[3][1])} "
        )
        current = rounded[3]
    return d


def _curved_run_d(seg: np.ndarray, max_error: float, *, cubic: bool, line_epsilon: float) -> str:
    quadratic = _append_quadratic_run("", seg, max_error, line_epsilon=line_epsilon, collapse_straight=False)
    if not cubic:
        return quadratic
    return _append_cubic_run("", seg, max_error, line_epsilon=line_epsilon)


def _fit_path_d(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    forced_corners: np.ndarray | None = None,
) -> str:
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
        if _segment_is_straight(seg, epsilon) and _segment_length(seg) >= minimum_line_length(epsilon):
            d += f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
        else:
            d += _curved_run_d(seg, max_error, cubic=cubic, line_epsilon=epsilon)
    d += "Z"
    return d


def _simpler_curve_candidate(
    contour: np.ndarray,
    *,
    baseline_d: str,
    epsilon: float,
    max_error: float,
    cubic: bool,
    forced_corners: np.ndarray | None,
) -> str:
    best_d = baseline_d
    best_count = _path_command_count(best_d)
    for eps_factor, err_factor in (
        (1.5, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (1.0, 20.0),
        (10.0, 30.0),
    ):
        candidate = _fit_path_d(
            contour,
            epsilon=max(epsilon * eps_factor, epsilon + 0.5),
            max_error=max(max_error * err_factor, max_error + 0.75),
            cubic=cubic,
            forced_corners=forced_corners,
        )
        candidate_count = _path_command_count(candidate)
        if candidate_count >= best_count:
            continue
        if _path_area_residual(contour, candidate) > SIMPLE_CURVE_RESIDUAL_TOL:
            continue
        best_d = candidate
        best_count = candidate_count
    return best_d


def fit_path(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    forced_corners: np.ndarray | None = None,
    prefer_simple_curves: bool = False,
) -> Shape:
    """Corner-split the contour; emit lines for straight runs, Béziers otherwise.

    ``cubic=False`` (default) fits each curved run with inflection-free
    quadratics — robust against the quantization staircase. ``cubic=True`` fits
    curved runs with denoised, inflection-guarded cubics; see PATH_DENOISE_EPS.

    ``prefer_simple_curves`` keeps the normal fit as a baseline, then accepts a
    looser smooth-curve fit only when it reduces path commands and stays close to
    the same filled contour.
    """
    d = _fit_path_d(
        contour,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        forced_corners=forced_corners,
    )
    if prefer_simple_curves:
        d = _simpler_curve_candidate(
            contour,
            baseline_d=d,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            forced_corners=forced_corners,
        )
    return Shape("path", {"d": d})
