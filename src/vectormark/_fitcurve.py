"""Vendored cubic-Bézier curve fitting.

Adapted from volkerp/fitCurves (MIT), an implementation of
Philip J. Schneider, "An Algorithm for Automatically Fitting Digitized Curves",
Graphics Gems (1990). Returns a list of (4, 2) control-point arrays.
"""

from __future__ import annotations

import numpy as np

bezier = lambda ctrl, t: (  # noqa: E731 - cubic Bézier point at t
    (1 - t) ** 3 * ctrl[0]
    + 3 * (1 - t) ** 2 * t * ctrl[1]
    + 3 * (1 - t) * t ** 2 * ctrl[2]
    + t ** 3 * ctrl[3]
)


def fit_cubic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []
    left_t = _normalize(pts[1] - pts[0])
    right_t = _normalize(pts[-2] - pts[-1])
    return _fit_cubic(pts, left_t, right_t, max_error)


def _normalize(v):
    n = np.hypot(*v)
    return v / n if n else v


def _fit_cubic(pts, left_t, right_t, error):
    if len(pts) == 2:
        dist = np.hypot(*(pts[0] - pts[1])) / 3.0
        ctrl = np.array([pts[0], pts[0] + left_t * dist, pts[1] + right_t * dist, pts[1]])
        return [ctrl]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_bezier(pts, u, left_t, right_t)
    max_err, split = _compute_max_error(pts, ctrl, u)
    if max_err < error:
        return [ctrl]
    if max_err < error * error:
        for _ in range(20):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _generate_bezier(pts, u, left_t, right_t)
            max_err, split = _compute_max_error(pts, ctrl, u)
            if max_err < error:
                return [ctrl]
    center_t = _normalize(pts[split - 1] - pts[split + 1])
    left = _fit_cubic(pts[: split + 1], left_t, center_t, error)
    right = _fit_cubic(pts[split:], -center_t, right_t, error)
    return left + right


def _generate_bezier(pts, u, left_t, right_t):
    A = np.zeros((len(u), 2, 2))
    A[:, 0] = left_t * (3 * (1 - u) ** 2 * u)[:, None]
    A[:, 1] = right_t * (3 * (1 - u) * u ** 2)[:, None]
    c = np.zeros((2, 2))
    x = np.zeros(2)
    first, last = pts[0], pts[-1]
    for i, ui in enumerate(u):
        c[0, 0] += A[i, 0] @ A[i, 0]
        c[0, 1] += A[i, 0] @ A[i, 1]
        c[1, 0] = c[0, 1]
        c[1, 1] += A[i, 1] @ A[i, 1]
        tmp = pts[i] - bezier(np.array([first, first, last, last]), ui)
        x[0] += A[i, 0] @ tmp
        x[1] += A[i, 1] @ tmp
    det_c = c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    det_x0 = x[0] * c[1, 1] - c[0, 1] * x[1]
    det_x1 = c[0, 0] * x[1] - x[0] * c[1, 0]
    alpha_l = 0.0 if det_c == 0 else det_x0 / det_c
    alpha_r = 0.0 if det_c == 0 else det_x1 / det_c
    seg = np.hypot(*(first - last))
    if alpha_l < 1e-6 * seg or alpha_r < 1e-6 * seg:
        d = seg / 3.0
        return np.array([first, first + left_t * d, last + right_t * d, last])
    return np.array([first, first + left_t * alpha_l, last + right_t * alpha_r, last])


def _reparameterize(pts, u, ctrl):
    return np.array([_newton(p, uu, ctrl) for p, uu in zip(pts, u)])


def _newton(p, u, ctrl):
    d = bezier(ctrl, u) - p
    q1 = 3 * (ctrl[1] - ctrl[0]) * (1 - u) ** 2 + 6 * (ctrl[2] - ctrl[1]) * (1 - u) * u + 3 * (ctrl[3] - ctrl[2]) * u ** 2
    q2 = 6 * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) * (1 - u) + 6 * (ctrl[3] - 2 * ctrl[2] + ctrl[1]) * u
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom


def _chord_length_parameterize(pts):
    u = np.zeros(len(pts))
    u[1:] = np.cumsum(np.hypot(*np.diff(pts, axis=0).T))
    return u / u[-1] if u[-1] else u


def _compute_max_error(pts, ctrl, u):
    errs = np.array([np.hypot(*(bezier(ctrl, uu) - p)) ** 2 for p, uu in zip(pts, u)])
    split = int(errs.argmax())
    return float(errs[split]), max(1, min(split, len(pts) - 2))
