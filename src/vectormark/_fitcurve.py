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


def cbezier(ctrl, t):
    """Cubic Bézier point(s) at parameter t."""
    mt = 1 - t
    return (mt**3 * ctrl[0] + 3*mt**2*t * ctrl[1] + 3*mt*t**2 * ctrl[2] + t**3 * ctrl[3])


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def cubic_inflects(ctrl: np.ndarray) -> bool:
    """True iff the planar cubic changes curvature sign in (0,1) — an inflection / S-curve.
    Sampled sign-change of cross(B'(t), B''(t)); threshold scaled by chord² filters
    FP noise on near-linear cubics where |B''| ≈ 0."""
    ctrl = np.asarray(ctrl, float)
    a, b, c = ctrl[1] - ctrl[0], ctrl[2] - ctrl[1], ctrl[3] - ctrl[2]   # first differences
    t = np.linspace(0.02, 0.98, 64)[:, None]
    d1 = 3 * ((1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c)          # B'(t)
    d2 = 6 * ((1 - t) * (b - a) + t * (c - b))                          # B''(t)
    cross = _cross(d1, d2)
    # threshold by chord² — FP noise on near-linear cubics is O(ε·chord²), real
    # inflection cross products are O(chord²), so 1e-6 cleanly separates them
    chord_sq = max(float(np.dot(ctrl[-1] - ctrl[0], ctrl[-1] - ctrl[0])), 1.0)
    tol = chord_sq * 1e-6
    s = np.sign(cross[np.abs(cross) > tol]); s = s[s != 0]
    return bool(np.any(np.diff(s) != 0))


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.array([0.0, 0.0])


def _generate_cubic(pts, u, t0, t3):
    """Schneider least-squares: endpoints pinned, end tangents fixed; solve the two tangent
    magnitudes a0,a3 (Graphics Gems 1990). P1 = P0 + a0*t0, P2 = P3 + a3*t3."""
    P0, P3 = pts[0], pts[-1]
    b0 = (1 - u)**3; b1 = 3*(1 - u)**2*u; b2 = 3*(1 - u)*u**2; b3 = u**3
    A1 = t0[None, :] * b1[:, None]; A2 = t3[None, :] * b2[:, None]
    f = pts - (P0 * (b0 + b1)[:, None] + P3 * (b2 + b3)[:, None])
    c11 = float((A1 * A1).sum()); c12 = float((A1 * A2).sum()); c22 = float((A2 * A2).sum())
    x1 = float((f * A1).sum()); x2 = float((f * A2).sum())
    det = c11 * c22 - c12 * c12
    chord = float(np.hypot(*(P3 - P0)))
    if abs(det) < 1e-12:
        a0 = a3 = chord / 3.0
    else:
        a0 = (x1 * c22 - x2 * c12) / det
        a3 = (c11 * x2 - c12 * x1) / det
    # keep control points on the tangent rays (no negative/backward magnitudes)
    lo = chord * 1e-2
    a0 = a0 if a0 > lo else chord / 3.0
    a3 = a3 if a3 > lo else chord / 3.0
    return np.array([P0, P0 + a0 * t0, P3 + a3 * t3, P3])


def _max_error_cubic(pts, ctrl, u):
    dev = np.linalg.norm(cbezier(ctrl, u[:, None]) - pts, axis=1)
    i = int(np.argmax(dev))
    return float(dev[i]), i


def fit_cubic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return []
    t0 = _unit(pts[1] - pts[0]); t3 = _unit(pts[-2] - pts[-1])
    return _fit_cubic(pts, max_error, t0, t3)


def _fit_cubic(pts, error, t0, t3):
    if len(pts) == 2:
        P0, P3 = pts[0], pts[1]
        d = (P3 - P0) / 3.0
        return [np.array([P0, P0 + d, P3 - d, P3])]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_cubic(pts, u, t0, t3)
    max_err, split = _max_error_cubic(pts, ctrl, u)
    if max_err < error and not cubic_inflects(ctrl):
        return [ctrl]
    if max_err < error * error and not cubic_inflects(ctrl):
        # Newton reparameterize a few times before giving up (error close)
        for _ in range(8):
            u = np.array([_newton_cubic(p, uu, ctrl) for p, uu in zip(pts, u)])
            ctrl = _generate_cubic(pts, u, t0, t3)
            max_err, split = _max_error_cubic(pts, ctrl, u)
            if max_err < error and not cubic_inflects(ctrl):
                return [ctrl]
    if split <= 0 or split >= len(pts) - 1:
        split = len(pts) // 2
    # interior tangent at the split (centered difference) shared by both halves
    tm = _unit(pts[split + 1] - pts[split - 1])
    left = _fit_cubic(pts[: split + 1], error, t0, -tm)
    right = _fit_cubic(pts[split:], error, tm, t3)
    return left + right


def _newton_cubic(p, u, ctrl):
    mt = 1 - u
    d = cbezier(ctrl, u) - p
    q1 = 3*mt**2*(ctrl[1]-ctrl[0]) + 6*mt*u*(ctrl[2]-ctrl[1]) + 3*u**2*(ctrl[3]-ctrl[2])
    q2 = 6*mt*(ctrl[2]-2*ctrl[1]+ctrl[0]) + 6*u*(ctrl[3]-2*ctrl[2]+ctrl[1])
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom
