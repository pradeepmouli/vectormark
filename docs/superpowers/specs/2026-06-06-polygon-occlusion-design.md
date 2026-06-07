# Polygon Occlusion Reconstruction Design

**Status:** Approved (design) — pending implementation plan
**Date:** 2026-06-06
**Depends on:** the annulus / N-shape occlusion pass (`occlusion.py`: `reconstruct_scene`,
`_complete_member`, `_pair_constraint`, `_topo_order`, `stack_agreement`, `primitive_mask`,
`label_boundary`), the existing `rdp` simplifier and `"polygon"` shape kind (`fit.py`,
`emit.py`).

## Problem

The N-shape occlusion pass reconstructs overlapping **circles, ellipses, and annuli** as a
z-ordered stack of completed primitives. It cannot reconstruct **polygons**: a convex polygon
occluded by another shape returns a partial fragment, and the existing completers only fit curved
primitives, so the fragment falls back to a faithful — but un-idealized — per-region path/polygon
fit that still carries the occluder's bite.

This spec adds a **convex-polygon completer** so an occluded diamond, triangle, or hexagon is
recovered to its true geometry, and lets the *existing* orchestration place it in the global
z-stack alongside circles and annuli.

## Scope

A single, self-contained addition:

- **In:** general **convex** polygons (triangle, diamond, pentagon, hexagon, …) where every edge
  remains at least partially visible, so each supporting line can be fit. Mixed groups
  (polygon + circle/annulus) resolve through the existing orchestration unchanged.
- **Out (explicit non-goals):** concave polygons; polygons with an entire edge hidden under the
  occluder (graceful decline → faithful fallback); regularity / symmetry snapping of the recovered
  polygon; a distinct-coloured polygon-overlap "lens" (the lens stays circle-only — a diamond
  overlap resolves by paint order, the same-colour-wins / Olympic model). The Olympic weave still
  declines (cyclic over/under).

The safety contract is unchanged from the annulus pass: the consistency gate guarantees the pass
only ever replaces a group it re-renders faithfully; everything else degrades to today's behaviour.

## Core idea

A convex polygon has a property that mirrors "complete a circle from its visible arc": even when a
**corner** is hidden, it can be recovered as the **intersection of its two adjacent edge-lines** —
provided both edges stay partially visible. So we fit straight **supporting lines** to the visible
(own-boundary) edges and intersect consecutive lines, cyclically, to recover every vertex —
including the one behind the seam.

```
group member (own/seam boundary already available)
  → own polyline       label_boundary(region, others)        (reused as-is)
  → visible edges      rdp(own) → ordered supporting lines     (reuses existing rdp)
  → vertices           intersect consecutive lines (cyclic); the last↔first
                       intersection recovers the hidden corner
  → accept             ≥3 lines, residual ≤ tol, convex ring, ≤ max_vertices
  → ScenePrimitive("polygon", {points})  →  global z-stack (unchanged)
```

## Components

### 1. `complete_polygon`

```
complete_polygon(region, others, *, max_residual, max_vertices) -> dict | None
```

1. **Own/seam split.** `label_boundary(region, others)` returns the outer contour as `(x, y)`
   points plus a per-point `seam` flag. The own points (`~seam`) form a contiguous **open**
   polyline, broken where the occluder bites.
2. **Visible edges.** Run the existing `rdp` simplifier on the own polyline. Its *interior*
   vertices are true visible corners; its two *endpoints* sit against the seam and are **not** real
   corners (that is where the boundary vanishes under the occluder). Each segment between
   consecutive RDP vertices defines a **supporting line**, fit to that segment's own points.
3. **Corner recovery.** Intersect each consecutive pair of supporting lines. Interior intersections
   reproduce the visible corners; the **wrap-around** intersection of the last and first supporting
   line recovers the corner hidden behind the seam.
4. **Accept iff:** ≥ 3 supporting lines; each line's maximum point-to-line residual ≤
   `max_residual`; the recovered vertex ring is **convex** (all consecutive edge cross-products
   share one sign — a cheap, strong false-positive guard, analogous to the annulus concentricity
   check); and vertex count ≤ `max_vertices`. Otherwise return `None`.
5. **Returns** `{"kind": "polygon", "params": {"points": [(x, y), …]}}` — the same representation
   `recognize_polygon` and the emit layer already use. The points are ordered around the ring.

A degenerate fit (parallel adjacent supporting lines → no intersection, a curved fragment whose
"edges" fail the residual bar, or a non-convex recovered ring) returns `None`, so the member
declines and falls back to today's faithful per-region fit.

### 2. `_complete_member` dispatch

Polygon becomes the **last** dispatch arm:

```
annulus (region has a hole)  →  circle / ellipse (curved fit)  →  polygon (straight-edge fit)
```

Order matters: the curved fitters reject a polygon (a straight edge blows the circle/ellipse
residual), so polygon is the natural fallthrough. A ring still completes as an annulus; a disk as a
circle; only genuinely straight-edged fragments reach `complete_polygon`.

### 3. `primitive_mask` polygon case

```
if prim["kind"] == "polygon":
    pts = prim["params"]["points"]                  # (x, y)
    rc = np.array([(y, x) for x, y in pts])         # skimage wants (row, col)
    return skimage.draw.polygon2mask((h, w), rc)
```

`primitive_mask` is the single chokepoint the orchestration reasons through — `_pair_constraint`
(overlap ownership → over/under), `_topo_order`, and `stack_agreement` (the consistency gate) all
call it. Once the mask is correct, the entire N-shape pipeline handles polygons with no further
change. `skimage.draw` is imported alongside the existing skimage uses in `occlusion.py`.

### 4. Emit & orchestration: no change

- `emit.shape_to_svg` and `emit.shape_to_path_d` already render `"polygon"` (`<polygon points=…>`
  / `M…L…L…Z`). A convex polygon is a single subpath, so **no `fill-rule`** (unlike the annulus).
- The non-flatten render path in `pipeline.py` maps each `ScenePrimitive` kind through
  `shape_to_svg`, which already knows `polygon`; the `--flatten` path uses `shape_to_path_d`, also
  covered. No new emit element, no flatten machinery.
- `reconstruct_scene` is unchanged: it appends `ScenePrimitive("polygon", params, color, z)` like
  any other kind, and the inferred global paint order, gate, and topo-sort apply generically. No
  lens / snap is produced for polygons.

### 5. Constants

Reuse `_MAX_RESIDUAL` (line-fit tolerance) and add `_MAX_VERTICES = 8` (matching
`recognize_polygon`'s vertex cap).

## Testing

**Unit** (append to `tests/test_occlusion.py`)

- `complete_polygon` recovers the known vertices of a synthetic **disk-occluded diamond** within
  tolerance.
- `complete_polygon` **rejects** a plain disk fragment (curved boundary → line residual exceeds the
  bar) and **declines** a fragment with a whole edge hidden behind the occluder.
- the convexity guard rejects a recovered ring that is concave.
- `primitive_mask` polygon membership: a point inside the recovered ring is `True`, one outside is
  `False`.

**Acceptance** (`tests/test_acceptance_polygon.py`, render SSIM ≥ 0.95)

- synthetic **two overlapping diamonds** (each clipping the other's corner) → two `<polygon>`
  primitives with a consistent z-order.
- synthetic **diamond occluded by a disk** → one polygon + one circle in a single group, proving
  the orchestration is shape-agnostic.

Both acceptance cases are proven primarily by a **recovered-vertex check** — the reconstructed
polygon's corners lie within tolerance of the known fixture geometry. This is the real fingerprint:
the per-region fallback instead emits a path or a bite-distorted polygon whose vertices do not match
the true corners. SSIM is a faithful-render sanity floor.

**Regression (critical)**

- the full suite stays green: **Mastercard** (two circles + lens + mirror snap), **Daikonic** (no
  false occlusion), the **annulus** acceptance fixtures, and the **Olympic-weave decline** are all
  untouched.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| RDP endpoints (seam stubs) mistaken for real corners | endpoints are dropped; only interior RDP vertices + the wrap intersection are corners; the consistency gate validates the result |
| A whole edge hidden → wrong wrap intersection | residual + convexity accept gate, then the consistency gate scores it below `_GATE_AGREEMENT` → decline → faithful fallback |
| Completer fires on a curved fragment (disk/annulus) | straight-line residual bar fails on a curved boundary; polygon is the *last* dispatch arm, after the curved fitters |
| Concave / self-intersecting recovered ring | convexity guard (consistent cross-product sign) rejects it |
| Parallel adjacent supporting lines (no intersection) | degenerate intersection → `None` → decline |

## Acceptance summary

The feature is correct when: a synthetic occluded diamond reconstructs to a clean convex
`<polygon>` whose recovered corners match the known geometry; two overlapping diamonds and a
diamond-occluded-by-disk reconstruct with a consistent global z-order and re-render faithfully
(SSIM ≥ 0.95); and Mastercard, Daikonic, the annulus fixtures, and the Olympic-weave decline are
unchanged. The consistency gate guarantees any reconstruction the pass emits re-renders to match
the input.
