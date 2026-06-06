# Annulus Occlusion Reconstruction Design

**Status:** Approved (design) — pending implementation plan
**Date:** 2026-06-05
**Depends on:** the v1.1 occlusion pass (`occlusion.py`: `reconstruct_scene`, `complete_primitive`,
`label_boundary`, `stack_agreement`), `skimage` `CircleModel`, the even-odd path machinery.

## Problem

v1.1 occlusion reconstructs **exactly two overlapping disks** (Mastercard): two circle/ellipse
crescents plus an optional distinct-coloured lens. It is hard-gated at `len(completed) != 2` and
assumes a single, trivial paint order. Two real generalizations are missing:

1. **Annuli (rings).** A ring occluded by another shape returns a partial-annulus fragment. The
   completer only fits a single outer circle, so rings collapse — the inner hole is lost.
2. **More than two overlapping shapes.** Ring chains and stacks (the Olympic-style arrangement)
   produce adjacency groups of 3+, which the two-crescent gate rejects outright.

This spec adds **annulus completion** and **N-shape orchestration with an inferred global paint
order**, keeping a single global z-stack (no per-segment weave).

## Scope: what a global z-stack delivers (and refuses)

The consistency gate compares the reconstruction to the input raster. A single global paint order
can represent any scene whose pairwise over/under relations are **acyclic**. So:

- ✅ a ring occluded by a disk → recover the full annulus + the disk
- ✅ two overlapping rings → two annuli with a consistent over/under
- ✅ non-interlaced ring stacks / chains where one global order is consistent
- ✅ **Mastercard unchanged** (two circles + lens + mirror snap) — regression-critical
- ⚠️ the **truly interlaced Olympic logo is a weave** (cyclic over/under): no global order
  reproduces it, so the gate **declines and falls back** to faithful per-region fitting. This is a
  *negative* test, not a win. The woven render needs per-segment z (an explicit non-goal here).

This is the same safety contract as v1.1: the reconstruction can only ever replace a group it
re-renders faithfully; everything else degrades to today's behaviour.

## Core idea

Extend the layered-primitive scene with an **annulus** primitive and replace the two-crescent
special case with a general pipeline:

```
group (adjacency, ≥2 regions, ≥1 bite)
  → complete each member        (annulus | circle | ellipse) from own-boundary points
  → infer pairwise over/under   (whose seam lies inside the other's completed shape)
  → topological sort            → a global paint order, or DECLINE on a cycle
  → consistency gate (N-stack)  paint in z-order, compare to region colours ≥ 0.97
  → emit ScenePrimitives in z-order, or fall back
```

## Components

### 1. Annulus primitive

A new `ScenePrimitive` kind `"annulus"` with `params = {cx, cy, r_outer, r_inner}` (two concentric
circles).

- **Mask** (`primitive_mask`): `r_inner² ≤ (x−cx)² + (y−cy)² ≤ r_outer²`.
- **SVG emit:** a single `<path>` with two circle subpaths under `fill-rule="evenodd"` (outer ring
  + inner ring), built from the existing `_ellipse_path_d` quarter-arc helper. No new emit element
  — it reuses the even-odd path machinery the pipeline already uses for holes/counters.

### 2. Annulus completer (`complete_annulus`)

A ring fragment's `region_contours` yields an **outer** contour and an **inner (hole)** contour,
both partial arcs when occluded.

1. Label own-vs-seam on **both** contours. `label_boundary` gains a `contour_index` parameter so
   the inner ring is labelled against the same neighbour set as the outer.
2. Fit the **outer** circle from outer-own points and the **inner** circle from inner-own points,
   reusing the existing `CircleModel` residual + own-arc-span (`_own_arc_span_deg`) logic.
3. **Accept** iff: both fits are within `_MAX_RESIDUAL`; both own arcs span ≥ `_MIN_ARC_DEG`; the
   two centres coincide within `_CONCENTRIC_TOL`; and `r_inner < r_outer`. The two centres are then
   merged to their mean so the emitted annulus is exactly concentric.
4. Returns `{"kind": "annulus", "params": {cx, cy, r_outer, r_inner}}`, else `None`.

A ring gives the completer *two* own arcs, over-constraining the fit; the concentricity check is a
cheap, strong false-positive guard before the consistency gate even runs.

### 3. N-shape orchestration + z-order DAG

Replaces the `len(completed) != 2` gate.

1. **Complete every member** of the adjacency group that has a bite (annulus first, then the
   existing circle/ellipse completer). Collect the `N` successes.
2. **Infer pairwise over/under** for each overlapping pair `(A, B)`: if `B`'s seam points lie
   inside `A`'s completed primitive, then **A occludes B** → constraint "paint B before A". This
   reuses the exact seam-inside test `complete_primitive` already performs for self-consistency.
3. **Topologically sort** the constraint DAG → a global paint order, assigning `z = 0..N-1`.
   - A **cycle** (mutual occlusion = a weave) ⇒ no valid order ⇒ **decline**.
   - A pair with **no constraint** (they don't actually overlap) is fine — topo-sort picks any
     order and the consistency gate validates it.
4. **Decline** (regions stay in `remaining`) if any member fails completion, the DAG is cyclic, or
   the gate scores `< _GATE_AGREEMENT`.

### 4. Consistency gate (N-stack)

`stack_agreement` generalizes to paint `N` primitives of any kind (circle / ellipse / annulus) in
`z`-order into a colour-label image and compare, over `region_union | painted_any`, to each
region's colour. The `≥ 0.97` bar is unchanged; `primitive_mask` gains the annulus case.

### 5. Mastercard / regression preservation

The two-circle-with-distinct-coloured-lens case is retained **inside** the general path: when
`N == 2`, both primitives are circles, and a leftover distinct-coloured intersection region exists,
keep the existing lens emit (`intersection_lens_d`) and `_snap_pair` mirror snap. For `N > 2` or any
annulus member, no lens/snap is produced (no motivating case — YAGNI).

## Output & flatten

- Completed annuli → an even-odd `<path>` (two concentric circle subpaths); z-order is the body
  paint order. `--flatten` already bakes overlapping z-ordered paths correctly (later fills cover
  earlier), so no new flatten machinery.
- Circles/ellipses and the circle lens are unchanged.

## Scope boundaries (explicit non-goals)

- **The weave** (per-segment / per-arc z) — interlaced Olympic stays a graceful decline.
- **Polygon completion** (overlapping diamonds) — a separate fast-follow spec that reuses this
  orchestration unchanged.
- **Distinct-coloured non-circle overlaps** — only circles keep the lens; annulus/annulus and
  annulus/disk overlaps resolve by paint order (same-colour-wins), which is the Olympic model.
- **Isolated (non-overlapping) rings** — the `group ≥ 2` trigger stays; a lone ring still goes
  through the existing multi-contour even-odd path fit, untouched.
- **N-fold / rotational symmetry snapping** of the ring arrangement — out of scope.

## Testing

**Unit**
- annulus `primitive_mask` ring membership; the even-odd `d` renders a hollow ring, not a disk
- `complete_annulus`: recovers known `cx, cy, r_outer, r_inner` from a synthetic occluded ring;
  rejects a single disk (no inner circle) and a non-concentric pair
- `label_boundary(contour_index=…)`: own-vs-seam correct on the inner contour
- pairwise over/under: seam-inside test gives the correct edge for a known ring-occluded-by-disk
- topological sort: valid DAG → an order; cyclic constraints → `None` (decline)
- `stack_agreement`: accepts a true N-annulus stack, rejects a degenerate one

**Acceptance (gating, render SSIM ≥ 0.95)**
- synthetic **ring + disk** → annulus + circle, correct z
- synthetic **two overlapping rings** → two annuli, consistent z
- synthetic **consistent-order 3-ring stack** → three annuli

**Negative / regression (critical)**
- **interlaced Olympic** (synthetic woven) → gate declines; regions fall back and still render
  faithfully; no crash, no bogus annuli
- **Mastercard** → unchanged (two circles + lens + mirror snap)
- **Daikonic** → unchanged (no false occlusion firing)

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Inner/outer own points mislabelled (wrong concentric fit) | per-contour `label_boundary`; concentricity + `r_inner < r_outer` accept gate; consistency gate |
| Weave fed in (cyclic over/under) | topological sort detects the cycle → decline → fall back |
| N-stack z-order wrong | good-continuation pairwise constraints + consistency gate validates the chosen order |
| Annulus completer fires on a plain disk | requires a real inner own arc spanning `_MIN_ARC_DEG`; else returns `None` |
| Mastercard regression from the rewrite | two-circle + lens + snap retained as a guarded case inside the general path; explicit regression test |

## Acceptance summary

The feature is correct when: a synthetic ring-occluded-by-disk and overlapping-ring fixtures
reconstruct to clean annulus primitives that re-render faithfully; an interlaced Olympic fixture is
safely declined and falls back; and Mastercard and Daikonic are unchanged. The consistency gate
guarantees any reconstruction it emits re-renders to match the input.
