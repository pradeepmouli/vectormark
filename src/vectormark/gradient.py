# SPDX-License-Identifier: MIT
"""Gradient detection: recover a gradient's footprint from quantized bands, fit a
linear/radial colour model against the original image, and emit one gradient-filled
shape when it re-renders faithfully (see
docs/superpowers/specs/2026-06-06-gradient-handling-design.md)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .color import srgb_to_oklab
from .occlusion import region_adjacency
from .types import Region

_MIN_BANDS = 3
_RAMP_TOL = 0.06          # max OKLab distance of a band colour from the fitted ramp line
_GATE_DELTA_E = 0.05      # mean OKLab ΔE bar: a fitted model must re-render within this to be accepted


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


def _ramp_fit(colors_oklab: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Fit a line to OKLab colours via the principal axis. Returns
    (unit_axis, max_residual, projections), or (None, inf, None) if degenerate/flat."""
    axis = _principal_axis(colors_oklab, eps=1e-6)
    if axis is None:
        return None, np.inf, None
    centred = colors_oklab - colors_oklab.mean(axis=0)
    projs = centred @ axis
    resid = float(np.linalg.norm(centred - np.outer(projs, axis), axis=1).max())
    return axis, resid, projs


def _is_ramp(colors_oklab: np.ndarray) -> bool:
    """True if >=3 colours and they lie on a single line in OKLab within _RAMP_TOL."""
    if len(colors_oklab) < _MIN_BANDS:
        return False
    _, resid, _ = _ramp_fit(colors_oklab)
    return resid <= _RAMP_TOL


def _is_strict_ramp(members: list[Region]) -> bool:
    """True if the members form a ramp AND all band colours project to distinct positions
    along the principal axis (monotone, no repeated colours)."""
    oklab = _hex_to_oklab([m.color_hex for m in members])
    axis, resid, projs = _ramp_fit(oklab)
    if axis is None or resid > _RAMP_TOL:
        return False
    # round(8) is a float-noise-tolerant *exact*-duplicate guard (catches palindromes like
    # green-blue-green). Near-duplicates that slip through are caught downstream by the
    # fit_gradient consistency gate: a folded colour sequence won't re-render monotonically.
    return len(np.unique(projs.round(8))) == len(members)


def _trim_to_ramp(members: list[Region], adj: dict) -> list[Region] | None:
    """Given a connected set of regions, trim boundary nodes that break the ramp property
    until either the remainder forms a strict ramp (return it) or fewer than _MIN_BANDS remain.
    Only degree<=1 (leaf) nodes are trimmable; a non-leaf intruder — e.g. a flat region
    bordering multiple ramp bands — causes the whole component to be rejected (the
    fit_gradient ΔE gate is the downstream backstop)."""
    current = list(members)
    while len(current) >= _MIN_BANDS:
        if _is_strict_ramp(current):
            return current
        # Find leaf nodes (degree ≤ 1 within the current subset)
        cur_labels = {m.label for m in current}
        leaves = [m for m in current
                  if len(adj[m.label] & cur_labels) <= 1]
        if not leaves:
            break                                           # no leaf to trim -> give up
        # Remove the leaf that least fits the ramp (highest residual after removal)
        best_trim = None
        best_score = np.inf
        for leaf in leaves:
            trial = [m for m in current if m.label != leaf.label]
            if len(trial) < _MIN_BANDS:
                continue
            _, score, _ = _ramp_fit(_hex_to_oklab([m.color_hex for m in trial]))
            if score < best_score:                          # strict `<` keeps the first
                best_score, best_trim = score, trial        # minimal-score leaf; deterministic
                                                            # given stable input `regions` order
        if best_trim is None:
            break
        current = best_trim
    return None


def _ramp_groups(regions: list[Region]) -> list[list[Region]]:
    """Connected groups of >=3 adjacent regions whose flat colours form an OKLab ramp.
    If a connected component's full colour set doesn't form a ramp, boundary nodes are
    trimmed iteratively until a ramp is found or the component is too small."""
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
            stack.extend(sorted(adj[lab] - seen))
        members = [by_label[l] for l in comp]
        if len(members) < _MIN_BANDS:
            continue
        ramp = _trim_to_ramp(members, adj)
        if ramp is not None:
            groups.append(ramp)
    return groups


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fit_stops(t: np.ndarray, rgb: np.ndarray, k: int = 7) -> list[tuple[float, str]]:
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


def _interp_stops_rgb(t: np.ndarray, stops: list[tuple[float, str]]) -> np.ndarray:
    offs = np.array([o for o, _ in stops])
    cols = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for _, c in stops], float)
    return np.column_stack([np.interp(t, offs, cols[:, ch]) for ch in range(3)])


def _model_t(model: dict, pts: np.ndarray) -> np.ndarray:
    g = model["geometry"]
    if model["kind"] == "linear":
        d = np.array([g["x2"] - g["x1"], g["y2"] - g["y1"]], float)
        L = float(d @ d) or 1.0
        return (((pts[:, 0] - g["x1"]) * d[0] + (pts[:, 1] - g["y1"]) * d[1]) / L).clip(0, 1)
    r = np.hypot(pts[:, 0] - g["cx"], pts[:, 1] - g["cy"]) / (g["r"] or 1.0)
    return r.clip(0, 1)


def _per_pixel_delta_e(model: dict, ys: np.ndarray, xs: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
    """Per-pixel OKLab ΔE between the model's rendered colour and the actual pixel at (ys, xs)."""
    pts = np.column_stack([xs, ys]).astype(float)
    rendered = _interp_stops_rgb(_model_t(model, pts), model["stops"])
    truth = rgb_image[ys, xs].astype(float)
    return np.linalg.norm(srgb_to_oklab(rendered / 255.0) - srgb_to_oklab(truth / 255.0), axis=1)


def _agreement_delta_e(model: dict, mask: np.ndarray, rgb_image: np.ndarray) -> float:
    """Mean OKLab ΔE between the rendered model and the original over the footprint."""
    ys, xs = np.where(mask)
    return float(_per_pixel_delta_e(model, ys, xs, rgb_image).mean())


def fit_gradient(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Fit a linear (then radial) gradient to the original pixels under `mask`. Returns
    the model only if it re-renders within _GATE_DELTA_E, else None."""
    ys, xs = np.where(mask)
    if len(xs) < 3 * _MIN_BANDS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    rgb = rgb_image[ys, xs].astype(float)
    oklab = srgb_to_oklab(rgb / 255.0)
    best = None
    for fit in (_fit_linear, _fit_radial):
        model = fit(pts, oklab, rgb)
        if model is None:
            continue
        de = _agreement_delta_e(model, mask, rgb_image)
        if de <= _GATE_DELTA_E and (best is None or de < best[0]):
            best = (de, model)
    return best[1] if best is not None else None


def _expand_footprint(model: dict, mask: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
    """Grow `mask` to include pixels outside it whose model-predicted colour matches
    the actual pixel within _GATE_DELTA_E. This recovers pixels that were classified
    as 'background' by the segmenter but are genuinely part of the gradient (the
    most-common border colour is sometimes the end of a full-canvas gradient rather
    than a neutral plate).

    Absorption is bounded to the connected component(s) of matching pixels that touch
    the original band mask, so a disconnected same-colour patch elsewhere is never
    swallowed. (A degenerate full-canvas gradient whose endpoint stop ≈ the page
    colour can still bridge across the page — but the render stays correct because
    only model-matching pixels are absorbed.)"""
    outside = ~mask
    if not outside.any():
        return mask
    oy, ox = np.where(outside)
    per_pixel_de = _per_pixel_delta_e(model, oy, ox, rgb_image)
    good = per_pixel_de <= _GATE_DELTA_E
    if not good.any():
        return mask
    match = mask.copy()
    match[oy[good], ox[good]] = True
    labels, _ = ndi.label(match)                 # 4-connectivity by default
    keep = set(labels[mask].tolist()) - {0}      # components overlapping the band mask
    return np.isin(labels, list(keep))


def detect_gradients(
    regions: list[Region], rgb_image: np.ndarray
) -> tuple[list[tuple[Region, dict]], list[Region]]:
    """Group ramp bands, fit+gate a gradient per footprint, and return
    (accepted [(footprint_region, model)], remaining flat regions)."""
    fills: list[tuple[Region, dict]] = []
    consumed: set[int] = set()
    for group in _ramp_groups(regions):
        mask = np.zeros(rgb_image.shape[:2], bool)
        for m in group:
            mask |= m.mask
        model = fit_gradient(mask, rgb_image)
        if model is None:
            continue                                   # dissolve back into flat bands
        mask = _expand_footprint(model, mask, rgb_image)
        rep = max(group, key=lambda r: r.area)
        footprint = Region(label=rep.label, mask=mask, color_hex=rep.color_hex)
        fills.append((footprint, model))
        consumed.update(m.label for m in group)
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
