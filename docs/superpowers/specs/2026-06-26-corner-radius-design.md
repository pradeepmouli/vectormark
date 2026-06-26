# Per-Region Corner Radius — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-26
**Area:** `pipeline.py` (corner-radius measurement + threading)

## Problem

`_mark_corner_radius` picks **one** corner-fillet radius for the whole mark and the
fitters apply it to every shape's corners. That single value cannot reflect per-shape
corner geometry, so it over-rounds marks with sharp corners:

| mark | corner_radius | source | result |
|------|------|------|------|
| gdrive | 20.8 | a fillet measured on 2 band-like regions, applied to ALL facets | sharp facet corners over-rounded |
| dropbox | 38.5 | fallback `0.22 × median height` (no fillet measurable) | sharp boxes rounded |
| appstore | 37.4 | same fallback | correct only because it IS a rounded square |
| instagram | 5.4 | measured | fine |

The fallback fraction invents rounding where a shape has none (dropbox), and a fillet
measured on some regions is wrongly applied to the sharp corners of others (gdrive).
A single global value cannot distinguish "this shape has rounded corners" from "this
shape has sharp corners" — and a mark can contain both (App Store's rounded square plus
its sharp letter "A", which are separate regions).

## Goal

Each fitted shape's corner radius is **measured from its own contour**: sharp corners
→ 0 (crisp), rounded corners → the measured radius. gdrive facets and dropbox boxes go
crisp; appstore/instagram keep their rounding; nothing that is genuinely rounded
(daikonic's rounded-trapezoid bands) regresses.

## Non-goals (YAGNI)

- Per-*corner* radii within one region (one region with both a sharp and a rounded
  corner gets a single representative radius). Within-region corners are near-uniform
  in practice; App Store's mix is across separate regions, which per-region already handles.
- Changing how the fitters apply `corner_radius` (they already round a shape's corners
  by the value given — only the *source* of the value changes).
- Curvature- or fit-and-compare-based measurement (rejected in brainstorming: noise-
  sensitive / expensive; the geometric line-fit inset reuses proven code).

## Design

### `_region_corner_radius(region: Region) -> float`

One representative corner-fillet radius for a region, measured from its contour; `0.0`
for a sharp-cornered shape. Generalizes the existing `_band_fillet_radius` (which only
measured a band's straight-taper side) to any polygon corner, and replaces both it and
`_mark_corner_radius`.

Mechanism:
1. Take the region contour (`region_contours(region.mask)[0]`); polygon-approximate it
   to locate corner vertices.
2. For each vertex `v` with along-contour neighbours: take the contour points on the two
   edges adjacent to `v`, **excluding a margin around `v`** so the rounded transition
   does not bias the edge fit. Fit a line to each edge (least squares). **Guard:** if
   either edge's max residual exceeds a straightness tolerance, skip the vertex (not a
   clean polygon corner).
3. Intersect the two fitted edge-lines → the sharp-corner point `P`. Measure the
   **inset**: how far the actual contour falls short of `P` toward the shape interior
   (the fillet depth). **Guard:** accept the inset only if it is small relative to the
   adjacent edge lengths (a corner fillet, not a curved cap / a degenerate near-straight
   join).
4. The region radius is the **median** of the accepted insets. Sharp corners produce
   inset ≈ 0, so an all-sharp region's median is ≈ 0.
5. **Pad only when a fillet is detected:** add `_DEANTIALIAS_PAD` to the result only if
   the median inset exceeds a small epsilon (the de-antialiasing correction applies to a
   real rounded corner). A sharp region (median ≈ 0) returns exactly `0.0` — never
   `0 + pad`.
6. If no vertex yields an accepted inset (too few points, no clean corners) → `0.0`.

Determinism: fixed polygon-approximation tolerance, least-squares fits, median — no
randomness or time.

### Integration (`pipeline.py`)

- In `build_candidates`, where `select_geometry(region, …, corner_radius, …)` is called
  per region, compute the radius per region:
  `cr = opt.corner_radius if opt.corner_radius is not None else _region_corner_radius(region)`.
  Pass `cr` to that region's fit. (For a mirror pair, measure the canonical region.)
- `_render_body` no longer computes a per-component `corner_radius`; the per-component
  `_mark_corner_radius(comp, axis)` call is removed.
- Delete `_mark_corner_radius`, `_band_fillet_radius`, and `_CORNER_RADIUS_FRACTION`
  (now unused). Keep `_DEANTIALIAS_PAD`. Keep `Options.corner_radius` as the global
  manual override (when set, it is used for every region exactly as today).

### Constants

| name | value | role |
|------|------|------|
| `_DEANTIALIAS_PAD` | 2.0 (existing) | added to a *detected* fillet radius only |
| `_CORNER_STRAIGHT_TOL` | ~2.5 px | max edge-line residual for a vertex to count as a clean corner (mirrors the existing `_band_fillet_radius` 2.5) |
| `_CORNER_MAX_FILLET_FRAC` | ~0.25 | max inset as a fraction of edge length to count as a fillet (mirrors `_band_fillet_radius`'s `0.25 * height`) |
| `_CORNER_MIN_FILLET` | ~1.0 px | min median inset to treat a region as rounded (below → 0, sharp, no pad) |

Starting values; corpus-validated before merge.

## Testing

- **Units:** a sharp square contour → `0.0`; a rounded-corner square (fillet radius r) →
  ≈ r (within a small tolerance); a rounded trapezoid → its measured fillet; a
  too-small/degenerate contour → `0.0`.
- **Pad rule:** a sharp square returns exactly `0.0` (asserts the pad is not added to a
  zero inset); a rounded square returns `measured + _DEANTIALIAS_PAD`.
- **Corpus:** gdrive facets and dropbox boxes idealize with sharp corners (no fillet);
  appstore/instagram keep their rounded corners; **daikonic's rounded-trapezoid bands
  still render rounded** (existing `test_acceptance_daikonic` is the parity guard); no
  other corpus mark regresses. Visual spot-check of gdrive/dropbox (crisp) and
  appstore/instagram (rounded).
- **Parity:** full suite green; `test_acceptance_daikonic` (structure + exact symmetry +
  render-ΔE) unchanged.

## Risks

- **Measurement accuracy on noisy/rasterized contours** — mitigated by the straightness
  guard and the margin-around-vertex exclusion; corpus-validate the thresholds.
- **Daikonic regression** — its bands are rounded trapezoids; if the general measurement
  under-measures their fillet they would go too sharp. The daikonic acceptance test
  guards this; tune `_CORNER_*` thresholds if it trips.
- **A genuinely-rounded shape `_band_fillet_radius` used to catch via the axis path** —
  the general method must measure it without the axis; covered by the daikonic and
  appstore/instagram corpus checks.
