"""Convex-only (quadratic-Bézier) curve fitting.

A quadratic Bézier is an affine image of the parabola ``t -> (t, t**2)``: its
second derivative is constant, so its curvature never changes sign. It is
therefore *inflection-free* — it cannot wobble into an S-curve the way a fitted
cubic can when it chases anti-aliasing noise. We fit each curved run with one or
more quadratics (subdividing on error), so every emitted free arc is convex.

Structure follows Schneider's cubic fitter (Graphics Gems, 1990) — chord-length
parameterize, solve for the single interior control point by least squares,
Newton-reparameterize, recurse on the worst point — but with the quadratic basis,
which has one free control point instead of two. Returns ``(3, 2)`` arrays.
"""

from __future__ import annotations

import numpy as np

qbezier = lambda ctrl, t: (  # noqa: E731 - quadratic Bézier point at t
    (1 - t) ** 2 * ctrl[0]
    + 2 * (1 - t) * t * ctrl[1]
    + t ** 2 * ctrl[2]
)


def fit_quadratic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []
    return _fit_quadratic(pts, max_error)


def _fit_quadratic(pts, error):
    if len(pts) == 2:
        return [np.array([pts[0], (pts[0] + pts[1]) / 2.0, pts[1]])]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_quadratic(pts, u)
    max_err, split = _compute_max_error(pts, ctrl, u)
    if max_err < error:
        return [ctrl]
    if max_err < error * error:
        for _ in range(20):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _generate_quadratic(pts, u)
            max_err, split = _compute_max_error(pts, ctrl, u)
            if max_err < error:
                return [ctrl]
    left = _fit_quadratic(pts[: split + 1], error)
    right = _fit_quadratic(pts[split:], error)
    return left + right


def _generate_quadratic(pts, u):
    """Closed-form least-squares interior control point P1 (endpoints pinned).

    For B(u) = (1-u)^2 P0 + 2(1-u)u P1 + u^2 P2 the only unknown is P1; the
    residual is linear in P1 with weight w = 2(1-u)u, so P1 minimizing
    sum |B(u_i) - p_i|^2 is sum(w_i r_i) / sum(w_i^2), per axis.
    """
    first, last = pts[0], pts[-1]
    w = 2 * (1 - u) * u
    rhs = pts - ((1 - u) ** 2)[:, None] * first - (u ** 2)[:, None] * last
    denom = float((w * w).sum())
    if denom < 1e-12:
        p1 = (first + last) / 2.0
    else:
        p1 = (w[:, None] * rhs).sum(axis=0) / denom
    return np.array([first, p1, last])


def _reparameterize(pts, u, ctrl):
    return np.array([_newton(p, uu, ctrl) for p, uu in zip(pts, u)])


def _newton(p, u, ctrl):
    d = qbezier(ctrl, u) - p
    q1 = 2 * (1 - u) * (ctrl[1] - ctrl[0]) + 2 * u * (ctrl[2] - ctrl[1])
    q2 = 2 * (ctrl[2] - 2 * ctrl[1] + ctrl[0])
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom


def _chord_length_parameterize(pts):
    u = np.zeros(len(pts))
    u[1:] = np.cumsum(np.hypot(*np.diff(pts, axis=0).T))
    return u / u[-1] if u[-1] else u


def _compute_max_error(pts, ctrl, u):
    errs = np.array([np.hypot(*(qbezier(ctrl, uu) - p)) ** 2 for p, uu in zip(pts, u)])
    split = int(errs.argmax())
    return float(errs[split]), max(1, min(split, len(pts) - 2))
