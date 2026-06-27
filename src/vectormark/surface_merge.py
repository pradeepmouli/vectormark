# SPDX-License-Identifier: MIT
"""Fill-informed surface merge: two adjacent shapes are one surface only when the source
has NO hard edge across their shared border (a within-gradient band seam) and the union
fits a parametric gradient. Boundaries come from clean masks; this decides which masks
are one surface, never how to draw an edge."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .candidate import Fill, LinearGradientFill, RadialGradientFill
from .color import srgb_to_oklab
from .fill_fit import fit_fill
from .gradient import _interp_stops_rgb, _model_t
from .types import Region

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _kind(fill: Fill) -> str | None:
    if isinstance(fill, LinearGradientFill):
        return "linear"
    if isinstance(fill, RadialGradientFill):
        return "radial"
    return None


def _model(fill: Fill) -> dict:
    return {"kind": _kind(fill), "geometry": fill.geometry, "stops": fill.stops}


def seam_pairs(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray):
    """Colors of every 4-adjacent pixel pair straddling the A|B boundary.

    Returns (colors_a, colors_b), each (N,3) uint8 — colors_a[k] in mask_a is 4-adjacent
    to colors_b[k] in mask_b."""
    cols_a, cols_b = [], []
    H, W = mask_a.shape
    for dy, dx in _NEIGHBORS:
        # b_shift[y,x] = mask_b[y+dy, x+dx]; valid region trims the wrapped edge.
        b_shift = np.zeros_like(mask_b)
        ys = slice(max(0, -dy), H - max(0, dy))
        xs = slice(max(0, -dx), W - max(0, dx))
        ys2 = slice(max(0, dy), H - max(0, -dy))
        xs2 = slice(max(0, dx), W - max(0, -dx))
        b_shift[ys, xs] = mask_b[ys2, xs2]
        border = mask_a & b_shift
        if not border.any():
            continue
        ay, ax = np.where(border)
        cols_a.append(rgb[ay, ax])
        cols_b.append(rgb[ay + dy, ax + dx])
    if not cols_a:
        return np.empty((0, 3), np.uint8), np.empty((0, 3), np.uint8)
    return np.concatenate(cols_a), np.concatenate(cols_b)


def seam_is_soft(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray,
                 *, edge_de: float = 0.06) -> bool:
    """(A) True iff the masks are 4-adjacent and the source color steps smoothly across
    their shared border (median straddling-pair OKLab ΔE < edge_de). A real object edge —
    even one whose two sides are color-similar — spikes above edge_de."""
    ca, cb = seam_pairs(mask_a, mask_b, rgb)
    if len(ca) == 0:
        return False
    de = np.linalg.norm(srgb_to_oklab(ca / 255.0) - srgb_to_oklab(cb / 255.0), axis=1)
    return float(np.median(de)) < edge_de


def seam_band(mask_a: np.ndarray, mask_b: np.ndarray, *, width: int = 2):
    """(ys, xs) of mask_a pixels within `width` px of mask_b — the A-side of the seam."""
    band = mask_a & ndimage.binary_dilation(mask_b, iterations=width)
    return np.where(band)


def _rendered_oklab(model: dict, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    pts = np.column_stack([xs, ys]).astype(float)
    return srgb_to_oklab(_interp_stops_rgb(_model_t(model, pts), model["stops"]) / 255.0)


def gradients_continuous(fill_a: Fill, mask_a: np.ndarray, fill_b: Fill, mask_b: np.ndarray,
                         *, seam_de: float = 0.045) -> bool:
    """(B) True iff both fills are gradients over adjacent masks whose models render to
    agreeing colours across the shared seam (mean OKLab ΔE < seam_de) — one gradient's
    colour at the seam matches the other's. False if either fill is flat (A handles that)."""
    if _kind(fill_a) is None or _kind(fill_b) is None:
        return False
    ys_a, xs_a = seam_band(mask_a, mask_b)
    ys_b, xs_b = seam_band(mask_b, mask_a)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return False                                     # not adjacent
    ma, mb = _model(fill_a), _model(fill_b)
    de1 = np.linalg.norm(_rendered_oklab(ma, ys_b, xs_b) - _rendered_oklab(mb, ys_b, xs_b), axis=1).mean()
    de2 = np.linalg.norm(_rendered_oklab(mb, ys_a, xs_a) - _rendered_oklab(ma, ys_a, xs_a), axis=1).mean()
    return max(float(de1), float(de2)) < seam_de


_GRADIENT = (LinearGradientFill, RadialGradientFill)


def merge_surfaces(filled: list[tuple[Region, Fill]], rgb: np.ndarray, *,
                   seam_de: float = 0.045, edge_de: float = 0.06) -> list[tuple[Region, Fill]]:
    """Fixed-point hybrid merge of adjacent surfaces. (B) both gradients -> merge when one's
    colour matches the other's at the seam (gradients_continuous, seam_de). (A) at least one
    flat (a narrow region that devolved) -> merge when the source has no edge across the seam
    (seam_is_soft, edge_de). Either path also requires the union to fit a parametric gradient,
    which becomes the merged fill. Keeps the larger member's label/color_hex. Deterministic:
    descending-area scan -> order-independent partition. A hard-bordered feature (the dot)
    never merges; two distinct flats whose union is not a gradient never merge."""
    surfaces = list(filled)
    merged = True
    while merged:
        merged = False
        surfaces.sort(key=lambda rf: rf[0].mask.sum(), reverse=True)
        for i in range(len(surfaces)):
            ri, fi = surfaces[i]
            for j in range(i + 1, len(surfaces)):
                rj, fj = surfaces[j]
                if isinstance(fi, _GRADIENT) and isinstance(fj, _GRADIENT):
                    ok = gradients_continuous(fi, ri.mask, fj, rj.mask, seam_de=seam_de)  # B
                else:
                    ok = seam_is_soft(ri.mask, rj.mask, rgb, edge_de=edge_de)             # A
                if not ok:
                    continue
                union = ri.mask | rj.mask
                rep = ri if ri.mask.sum() >= rj.mask.sum() else rj
                new_fill = fit_fill(union, rgb, flat_hex=rep.color_hex)
                if not isinstance(new_fill, _GRADIENT):
                    continue                              # union isn't a gradient: not a merge
                new_region = Region(label=rep.label, mask=union, color_hex=rep.color_hex)
                surfaces = ([s for k, s in enumerate(surfaces) if k not in (i, j)]
                            + [(new_region, new_fill)])
                merged = True
                break
            if merged:
                break
    return surfaces
