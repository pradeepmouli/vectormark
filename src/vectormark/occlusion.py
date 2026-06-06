# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import CircleModel, EllipseModel
from skimage.morphology import convex_hull_image

from .contour import region_contours
from .fit import Shape, _fmt
from .types import Axis, Region


@dataclass
class ScenePrimitive:
    """A completed shape that may be partially occluded by higher-z primitives."""
    kind: str                 # "circle" | "ellipse"
    params: dict
    color_hex: str
    z: int


def has_bite(mask: np.ndarray, *, max_solidity: float = 0.92) -> bool:
    """True when the region is non-convex enough to be a plausible occluded fragment
    (a crescent), i.e. its solidity (area / convex-hull area) is below the bar."""
    area = int(mask.sum())
    if area == 0:
        return False
    hull = int(convex_hull_image(mask).sum())
    return hull > 0 and (area / hull) < max_solidity


def label_boundary(
    region: Region, others: list[Region], *, reach: int = 2, contour_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (contour Nx2 as (x,y), seam_bool N) for the region's `contour_index`-th
    contour (0 = outer boundary, 1 = largest hole, ...). A contour point is a seam if
    any OTHER region's mask sits within `reach` px of it; else it is own boundary."""
    contours = region_contours(region.mask)
    if contour_index >= len(contours):
        return np.empty((0, 2)), np.empty((0,), bool)
    contour = contours[contour_index]
    if not others:
        return contour, np.zeros(len(contour), bool)
    near = np.zeros_like(region.mask)
    for o in others:
        near |= binary_dilation(o.mask, iterations=reach)
    h, w = region.mask.shape
    xs = np.clip(np.rint(contour[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(contour[:, 1]).astype(int), 0, h - 1)
    seam = near[ys, xs]
    return contour, seam


def region_adjacency(regions: list[Region]) -> dict[int, set[int]]:
    """label -> set of labels whose masks touch it (8-connectivity, 1px dilation)."""
    adj: dict[int, set[int]] = {r.label: set() for r in regions}
    dilated = {r.label: binary_dilation(r.mask) for r in regions}
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if (dilated[a.label] & b.mask).any():
                adj[a.label].add(b.label)
                adj[b.label].add(a.label)
    return adj


def _own_arc_span_deg(own_pts: np.ndarray, cx: float, cy: float) -> float:
    """Angular span (deg) of the own points about (cx, cy). Full circle -> ~360."""
    ang = np.sort(np.arctan2(own_pts[:, 1] - cy, own_pts[:, 0] - cx))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(np.degrees(2 * np.pi - gaps.max()))     # span covered = full minus largest gap


def _fit_candidate_pts(own: np.ndarray) -> np.ndarray:
    """Return the convex-hull vertices of `own` when feasible, else `own` itself.
    Using the convex hull isolates the outer perimeter, discarding inner concave arcs
    that arise when seam detection cannot mark the full interior boundary."""
    from scipy.spatial import ConvexHull, QhullError
    if len(own) < 4:
        return own
    try:
        hull = ConvexHull(own)
        return own[hull.vertices]
    except QhullError:
        return own


def _fit_circle(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle to the own-boundary points (convex hull, to drop concave inner
    arcs). Returns {"cx","cy","r"} or None if too few points, residual too large, or
    the own arc spans less than `min_arc_deg`."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    cm = CircleModel.from_estimate(fit_pts)
    if not cm or np.abs(cm.residuals(fit_pts)).max() > max_residual:
        return None
    cx, cy = float(cm.center[0]), float(cm.center[1])
    if _own_arc_span_deg(fit_pts, cx, cy) < min_arc_deg:
        return None
    return {"cx": cx, "cy": cy, "r": float(cm.radius)}


def complete_primitive(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle (then ellipse) to the OWN-boundary points, completing across the
    seam. Returns {"kind","params"} or None when the own arc can't constrain a fit.

    Fitting is performed on the convex hull of the own points so that inner concave
    arcs (which occur when the occluder's overlap zone is not passed as a neighbor
    region and thus not captured by seam detection) do not bias the fit."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    circ = _fit_circle(contour, seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    if circ is not None:
        cx, cy = circ["cx"], circ["cy"]
        seam_pts = np.asarray(contour, float)[seam]
        inside = len(seam_pts) == 0 or np.all(
            (seam_pts[:, 0] - cx) ** 2 + (seam_pts[:, 1] - cy) ** 2 <= (circ["r"] + max_residual) ** 2
        )
        if inside:
            return {"kind": "circle", "params": {"cx": cx, "cy": cy, "r": circ["r"]}}
    em = EllipseModel.from_estimate(fit_pts)
    if em and np.abs(em.residuals(fit_pts)).max() <= max_residual:
        xc, yc = float(em.center[0]), float(em.center[1])
        a, b = (float(v) for v in em.axis_lengths)
        if abs(em.theta) < 0.08 or abs(abs(em.theta) - np.pi) < 0.08:
            if _own_arc_span_deg(fit_pts, xc, yc) >= min_arc_deg:
                return {"kind": "ellipse", "params": {"cx": xc, "cy": yc, "rx": a, "ry": b}}
    return None


def intersection_lens_d(a: dict, b: dict) -> str | None:
    """SVG path (two A arcs) for the lens = intersection of circles a and b.
    Returns None if the circles don't properly overlap (disjoint or one contains
    the other). Params dicts use cx, cy, r."""
    ax, ay, ar = a["cx"], a["cy"], a["r"]
    bx, by, br = b["cx"], b["cy"], b["r"]
    dx, dy = bx - ax, by - ay
    dist = float(np.hypot(dx, dy))
    if dist <= abs(ar - br) or dist >= ar + br or dist == 0:
        return None                                       # nested or disjoint
    t = (dist * dist + ar * ar - br * br) / (2 * dist)
    h2 = ar * ar - t * t
    if h2 <= 0:
        return None
    hh = float(np.sqrt(h2))
    ux, uy = dx / dist, dy / dist                          # unit a->b
    mx, my = ax + t * ux, ay + t * uy                      # chord midpoint
    p1 = (mx - hh * (-uy), my - hh * ux)                   # one crossing
    p2 = (mx + hh * (-uy), my + hh * ux)                   # the other
    f = _fmt
    return (
        f"M{f(p1[0])} {f(p1[1])} "
        f"A{f(ar)} {f(ar)} 0 0 1 {f(p2[0])} {f(p2[1])} "
        f"A{f(br)} {f(br)} 0 0 1 {f(p1[0])} {f(p1[1])} Z"
    )


def primitive_mask(prim: dict, h: int, w: int) -> np.ndarray:
    """Boolean mask of a completed circle/ellipse on an (h, w) grid."""
    yy, xx = np.ogrid[:h, :w]
    p = prim["params"]
    if prim["kind"] == "circle":
        return (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2 <= p["r"] ** 2
    if prim["kind"] == "annulus":
        d2 = (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2
        return (d2 <= p["r_outer"] ** 2) & (d2 >= p["r_inner"] ** 2)
    return ((xx - p["cx"]) / p["rx"]) ** 2 + ((yy - p["cy"]) / p["ry"]) ** 2 <= 1.0


def stack_agreement(prims, lens, regions: list[Region], h: int, w: int) -> float:
    """Paint prims (then the lens) in z-order into a colour-label image and compare,
    over the union of the regions and any pixels painted by the stack, to each
    region's own colour. Returns [0, 1]."""
    BG = "\x00"
    painted = np.full((h, w), BG, dtype=object)
    painted_any = np.zeros((h, w), bool)
    for prim in sorted(prims, key=lambda p: p["z"]):
        m = primitive_mask(prim, h, w)
        painted[m] = prim["color"]
        painted_any |= m
    if lens is not None:
        a, b = (prims[lens["lens_of"][0]], prims[lens["lens_of"][1]])
        lm = primitive_mask(a, h, w) & primitive_mask(b, h, w)
        painted[lm] = lens["mask_color"]
        painted_any |= lm
    truth = np.full((h, w), BG, dtype=object)
    region_union = np.zeros((h, w), bool)
    for r in regions:
        truth[r.mask] = r.color_hex
        region_union |= r.mask
    compare_mask = region_union | painted_any
    if not compare_mask.any():
        return 0.0
    return float((painted[compare_mask] == truth[compare_mask]).mean())


_MAX_RESIDUAL = 1.6
_MIN_ARC_DEG = 110.0
_GATE_AGREEMENT = 0.97


def _snap_pair(p_left: dict, p_right: dict, axis_x: float) -> tuple[dict, dict]:
    """Force two completed circles to be exact mirror images about x = axis_x."""
    r = (p_left["params"]["r"] + p_right["params"]["r"]) / 2
    cy = (p_left["params"]["cy"] + p_right["params"]["cy"]) / 2
    off = (abs(p_left["params"]["cx"] - axis_x) + abs(p_right["params"]["cx"] - axis_x)) / 2
    left = {"kind": "circle", "params": {"cx": axis_x - off, "cy": cy, "r": r}}
    right = {"kind": "circle", "params": {"cx": axis_x + off, "cy": cy, "r": r}}
    return left, right


def reconstruct_scene(
    regions: list[Region], axis: Axis | None, shape_hw: tuple[int, int]
) -> tuple[list, list[Region]]:
    """Return (reconstructed, remaining). `reconstructed` mixes ScenePrimitive and
    lens Shape objects in paint order; `remaining` are regions to fit the old way.
    Only adjacent groups that pass the consistency gate are reconstructed."""
    h, w = shape_hw
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    reconstructed: list = []
    consumed: set[int] = set()

    for r in regions:
        if r.label in consumed or not has_bite(r.mask):
            continue
        # transitive connected component (crescents touch only through the lens)
        group: set[int] = set()
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in group:
                continue
            group.add(lab)
            stack.extend(adj[lab] - group)
        group_regions = [by_label[l] for l in sorted(group) if l not in consumed]
        if len(group_regions) < 2:
            continue

        completed: list[tuple[Region, dict]] = []
        for gr in group_regions:
            if not has_bite(gr.mask):
                continue
            others = [o for o in group_regions if o.label != gr.label]
            contour, seam = label_boundary(gr, others)
            prim = complete_primitive(contour, seam, max_residual=_MAX_RESIDUAL, min_arc_deg=_MIN_ARC_DEG)
            if prim is not None:
                completed.append((gr, prim))
        if len(completed) != 2:
            continue                                      # v1 handles the two-disk case

        (ra, pa), (rb, pb) = completed
        # v1 scope: a clean two-disk overlap is EXACTLY the two crescents plus at most
        # one lens (the distinct-coloured intersection). A larger connected component
        # (chains, 3+ overlaps) is out of scope — decline rather than risk dropping a
        # member or mis-picking the lens. (z-order between the two disks needs no
        # disambiguation here: when a distinct-coloured lens exists it is painted on
        # top and covers the intersection, so disk paint-order is moot; equal-coloured
        # overlaps merge into one region upstream and never reach this two-crescent path.)
        non_completed = [g for g in group_regions if g.label not in {ra.label, rb.label}]
        if len(non_completed) > 1:
            continue
        if pa["params"]["cx"] > pb["params"]["cx"]:
            (ra, pa), (rb, pb) = (rb, pb), (ra, pa)
        if axis is not None and pa["kind"] == pb["kind"] == "circle":
            snapped_l, snapped_r = _snap_pair(pa, pb, axis.x)
            pa["params"], pb["params"] = snapped_l["params"], snapped_r["params"]

        prims = [
            {"kind": pa["kind"], "params": pa["params"], "color": ra.color_hex, "z": 0},
            {"kind": pb["kind"], "params": pb["params"], "color": rb.color_hex, "z": 1},
        ]
        lens_region = non_completed[0] if non_completed else None
        lens = None
        if lens_region is not None and pa["kind"] == pb["kind"] == "circle":
            lens = {"mask_color": lens_region.color_hex, "lens_of": (0, 1)}

        if stack_agreement(prims, lens, group_regions, h, w) < _GATE_AGREEMENT:
            continue                                      # reject -> regions stay in `remaining`

        reconstructed.append(ScenePrimitive(pa["kind"], pa["params"], ra.color_hex, 0))
        reconstructed.append(ScenePrimitive(pb["kind"], pb["params"], rb.color_hex, 1))
        if lens is not None:
            d = intersection_lens_d(pa["params"], pb["params"])
            if d is not None:
                reconstructed.append(Shape("path", {"d": d, "color_hex": lens_region.color_hex, "z": 2}))
        # consume ONLY what was reconstructed (the guard guarantees the group has no
        # other members, but be explicit so a stale group can never drop a region)
        consumed.update({ra.label, rb.label})
        if lens_region is not None:
            consumed.add(lens_region.label)

    remaining = [r for r in regions if r.label not in consumed]
    return reconstructed, remaining
