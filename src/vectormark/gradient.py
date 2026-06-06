# SPDX-License-Identifier: MIT
"""Gradient detection: recover a gradient's footprint from quantized bands, fit a
linear/radial colour model against the original image, and emit one gradient-filled
shape when it re-renders faithfully (see
docs/superpowers/specs/2026-06-06-gradient-handling-design.md)."""

from __future__ import annotations

import numpy as np

from .color import srgb_to_oklab
from .occlusion import region_adjacency
from .types import Region

_MIN_BANDS = 3
_RAMP_TOL = 0.06          # max OKLab distance of a band colour from the fitted ramp line
_GATE_DELTA_E = 0.05      # mean OKLab ΔE bar (consumed by fit_gradient in a later task)


def _hex_to_oklab(hex_colors: list[str]) -> np.ndarray:
    rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_colors],
        float,
    )
    return srgb_to_oklab(rgb / 255.0)


def _principal_axis(vectors: np.ndarray, *, eps: float) -> np.ndarray | None:
    """Mean-centre `vectors` and return the unit principal direction (top SVD right-
    singular vector), or None if the spread is below `eps` (degenerate/flat)."""
    centred = vectors - vectors.mean(axis=0)
    if np.abs(centred).max() < eps:
        return None
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def _is_ramp(colors_oklab: np.ndarray) -> bool:
    """True if >=3 colours and they lie on a single line in OKLab within _RAMP_TOL."""
    if len(colors_oklab) < _MIN_BANDS:
        return False
    axis = _principal_axis(colors_oklab, eps=1e-6)
    if axis is None:
        return False                                   # all equal -> flat, not a ramp
    centred = colors_oklab - colors_oklab.mean(axis=0)
    proj = np.outer(centred @ axis, axis)
    resid = np.linalg.norm(centred - proj, axis=1).max()
    return resid <= _RAMP_TOL


def _ramp_groups(regions: list[Region]) -> list[list[Region]]:
    """Connected groups of >=3 adjacent regions whose flat colours form an OKLab ramp."""
    # Grows each full connected component, then tests if the whole thing is a ramp.
    # segment() excludes the background plate, so a gradient mark's bands neighbor only
    # each other -> this works. A gradient bordering another real shape is lumped in and
    # rejected (falls back to flat bands); edge-consistent growth would refine that later.
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    seen: set[int] = set()
    groups: list[list[Region]] = []
    for r in regions:
        if r.label in seen:
            continue
        # grow the connected component
        comp: list[int] = []
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in seen:
                continue
            seen.add(lab)
            comp.append(lab)
            stack.extend(adj[lab] - seen)
        members = [by_label[l] for l in comp]
        if len(members) < _MIN_BANDS:
            continue
        if _is_ramp(_hex_to_oklab([m.color_hex for m in members])):
            groups.append(members)
    return groups


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fit_stops(t: np.ndarray, rgb: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
    """Sample k evenly-spaced stops; each stop colour = median original RGB of pixels
    in its t-neighbourhood. `t` in [0,1], `rgb` is (N,3) uint8-ish."""
    edges = np.linspace(0.0, 1.0, k)
    half = 0.5 / (k - 1)
    stops: list[tuple[float, str]] = []
    for e in edges:
        sel = np.abs(t - e) <= half
        if not sel.any():
            sel = np.argsort(np.abs(t - e))[:32]       # nearest fallback
        stops.append((float(e), _rgb_to_hex(np.median(rgb[sel], axis=0))))
    return stops


def _stop_colors_oklab(stops: list[tuple[float, str]]) -> np.ndarray:
    return _hex_to_oklab([c for _, c in stops])


def _reduce_stops(stops: list[tuple[float, str]], *, max_delta_e: float) -> list[tuple[float, str]]:
    """Greedily drop interior stops whose removal keeps the piecewise-linear (in OKLab)
    reconstruction within max_delta_e of the full stop set."""
    kept = list(stops)
    changed = True
    while changed and len(kept) > 2:
        changed = False
        ref_off = np.array([o for o, _ in kept])
        ref_cols = _stop_colors_oklab(kept)
        for i in range(1, len(kept) - 1):
            trial = kept[:i] + kept[i + 1:]
            offs = np.array([o for o, _ in trial])
            cols = _stop_colors_oklab(trial)
            approx = np.column_stack([np.interp(ref_off, offs, cols[:, ch]) for ch in range(3)])
            if np.linalg.norm(approx - ref_cols, axis=1).max() <= max_delta_e:
                kept = trial
                changed = True
                break
    return kept


def _radial_spread(pts: np.ndarray, oklab: np.ndarray, c: np.ndarray, nbins: int = 16) -> float:
    """Mean within-bin colour variance when binning pixels by distance from c. Lower =
    more concentric (a better radial centre)."""
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = r.max()
    if rmax < 1e-6:
        return np.inf
    bins = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    total, count = 0.0, 0
    for b in range(nbins):
        sel = bins == b
        if sel.sum() >= 2:
            total += float(np.var(oklab[sel], axis=0).sum())
            count += 1
    return total / count if count else np.inf


def _fit_radial(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a radial gradient: estimate the centre as the centroid of the extreme along
    the principal colour axis (try both ends; keep the more concentric), then fit stops
    vs normalized radius. The centroid-of-extreme heuristic can yield a slightly-off
    centre for clipped or asymmetric footprints; the Task 6 consistency gate is the
    backstop that rejects fits that don't re-render faithfully."""
    axis = _principal_axis(oklab, eps=1e-8)
    if axis is None:
        return None
    s = (oklab - oklab.mean(axis=0)) @ axis             # 1-D colour coordinate
    order = np.argsort(s)
    k = max(3, len(s) // 10)
    best_c, best_spread = None, np.inf
    for idx in (order[-k:], order[:k]):                 # inner-extreme and outer-extreme clusters
        c = pts[idx].mean(axis=0)
        spread = _radial_spread(pts, oklab, c)
        if spread < best_spread:
            best_spread, best_c = spread, c
    if best_c is None:
        return None
    c = best_c
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = float(r.max())
    if rmax < 1e-6:
        return None
    tn = r / rmax
    stops = _reduce_stops(_fit_stops(tn, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "radial",
            "geometry": {"cx": float(c[0]), "cy": float(c[1]), "r": rmax},
            "stops": stops}


def _fit_linear(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a linear gradient. pts:(N,2) xy, oklab:(N,3), rgb:(N,3). Returns a model or
    None if the points don't span an axis."""
    A = np.column_stack([pts, np.ones(len(pts))])
    coef, *_ = np.linalg.lstsq(A, oklab, rcond=None)   # rows: [p, q, r] per channel column
    G = coef[:2, :].T                                   # (3 channels, 2) per-channel gradient
    if np.abs(G).max() < 1e-8:
        return None
    _, _, vt = np.linalg.svd(G, full_matrices=False)
    u = vt[0]                                           # unit axis direction
    proj = pts @ u
    t0, t1 = float(proj.min()), float(proj.max())
    if t1 - t0 < 1e-6:
        return None
    tn = (proj - t0) / (t1 - t0)
    mean = pts.mean(axis=0)
    mt = float(mean @ u)
    p1 = mean + (t0 - mt) * u
    p2 = mean + (t1 - mt) * u
    stops = _reduce_stops(_fit_stops(tn, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "linear",
            "geometry": {"x1": float(p1[0]), "y1": float(p1[1]),
                         "x2": float(p2[0]), "y2": float(p2[1])},
            "stops": stops}
