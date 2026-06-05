# Occlusion Reconstruction Design

**Status:** Approved (design) — pending implementation plan
**Date:** 2026-06-04
**Depends on:** the v1 pipeline (`segment` → `_fit_region` → emit), `shapely`, `skimage` `CircleModel`/`EllipseModel`

## Problem

v1 assumes **each flat color region is one complete shape**. That holds for stacked/nested
marks (Daikonic, Target) but breaks under **occlusion**: when shapes overlap, segmentation
returns only the *visible fragment*. Mastercard is not "a red crescent + a yellow crescent +
an orange lens"; it is **two full disks painted in z-order**, and the fragments are what
remains after the overlap. Real-logo testing confirmed this: Mastercard's circles collapse to
crescents (no primitive recognized) and the Olympic rings break entirely.

The reconstruction must **generalize** — a Mastercard-only special case is not worth building.
The first implementation targets circles + ellipses, but via a primitive-agnostic mechanism
that extends to polygons and annuli without re-architecting.

## Core idea: a layered-primitive scene

Model the image as a **z-ordered stack of opaque primitives; each region is the topmost
primitive visible there**. A region's contour is therefore a mix of:

- **own boundary** — where it borders background/transparency (its true edge), and
- **seam boundary** — where it borders another region (an occlusion edge).

The reconstruction:

1. Fit a primitive to the **own boundary only**, extrapolating *across* the seams → the
   completed shape.
2. Reconstruct the image as a **z-ordered stack** of completed primitives; each visible
   region is the topmost primitive there.
3. **Consistency gate:** re-render the stack and require it to reproduce the original
   segmentation; otherwise reject the reconstruction and fall back to per-region fitting.

Subtraction and union are two views of the same recovery: red crescent = `red ∖ yellow`
(subtraction) ≡ `red_crescent ∪ orange_lens = red_disk` (union); the lens = `red ∩ yellow`
(intersection). Recover the disks and every fragment is explained.

This is **additive and safe**: it can only ever replace a group it can re-render faithfully.
Marks with no occlusion (Daikonic's abutting bands) produce no completion and degrade exactly
to today's behavior.

## Architecture

A new pass `reconstruct_scene(regions, axis)` sits between `segment()` and the per-region fit
loop in `idealize()`:

```
segment → regions
        → reconstruct_scene(regions, axis)            # NEW
        → Scene (z-ordered: ScenePrimitives + leftover Regions + overlap shapes)
        → fit each (primitives already carry geometry; leftover Regions → existing _fit_region)
        → emit in z-order (+ existing axis-snap / <use> mirror / flatten)
```

### Data model (two small new types)

- `ScenePrimitive` — a completed shape that may be partially occluded:
  `kind` ("circle" | "ellipse"), `params` (dict), `color` (hex), `z` (int paint order).
- `Scene` — an ordered list whose entries are either a `ScenePrimitive` or a passed-through
  `Region`, plus any derived overlap shapes (e.g. a distinct-colored intersection lens).

## Components

1. **Adjacency graph.** Region pairs that touch (1px binary-dilation overlap). Connected
   adjacent groups are occlusion *candidates*; isolated regions pass straight through.

2. **Bite trigger.** Only attempt completion for a region whose contour has a **concave**
   segment (curves *into* the region) that is also a seam. A crescent's inner arc qualifies; a
   convex band does not. This is the primary guard against mistaking abutment (Daikonic) for
   occlusion.

3. **Boundary labeller.** Tag each contour point own vs seam by inspecting the label map in a
   1–2px neighborhood (seam = another region adjacent; own = background adjacent).

4. **Completer (circles/ellipses).** Fit `CircleModel`/`EllipseModel.from_estimate` to the
   **own-boundary points only**, extrapolating across seams. Accept iff (a) the fit residual is
   tight, (b) the own arc spans a minimum angle (enough to constrain the circle), and (c) the
   seam points lie inside/on the completed shape (consistent with being occluded). Otherwise
   return nothing and let the region fall through.

5. **Overlap + z-order.** `shapely` intersection of completed primitives. A distinct-colored
   region sitting at the intersection → emit it as its own **lens shape on top** (color taken
   from that region). Z-order from the good-continuation cue (the region whose boundary
   continues across a seam is in front), **disambiguated by the consistency gate** — render the
   few candidate orderings and keep the one that matches.

6. **Consistency gate.** Rasterize the reconstructed z-stack at region resolution and compare
   per-pixel to the original quantized labels; require ≥ ~98% agreement. Fail → discard the
   reconstruction for that group, fall back to per-region fitting. This makes false positives
   harmless and lets the bite trigger be liberal — the gate is the real arbiter.

## Output & flatten

- Completed primitives → existing `shape_to_svg` (`<circle>` / `<ellipse>`); z-order is the
  paint order in the body list.
- A distinct-colored intersection → a 2-arc lens `<path>` built from the shapely intersection,
  painted on top.
- The existing `--flatten` bakes the whole stack to non-overlapping paths by emitting each
  primitive as `shape_to_path_d` **in z-order** (later fills cover earlier — reproduces the
  image). No new output machinery.

## Symmetry integration (free)

Mastercard's two disks are a mirror pair about the detected vertical axis, and the central lens
straddles it. So reconstruction → recognize the two completed circles as a pair → emit **one
`<circle>` + `<use>` mirror + a symmetric lens** on top. The existing `_snap_to_axis`, pair
classification, and `<use>` mirroring apply to `ScenePrimitive`s unchanged.

## Testing

**Unit**
- adjacency graph from touching regions
- bite/concavity detection (crescent inner arc = concave; trapezoid = convex)
- own-vs-seam boundary labelling
- circle completion from a crescent's own arc (recovers known center/radius within tolerance)
- shapely intersection-lens geometry (two-arc path)
- consistency gate: accept on a true reconstruction, reject on a degenerate one

**Acceptance (gating)**
- rasterized **Mastercard** → two disks (as `<circle>` + `<use>`) + one lens; render
  SSIM ≥ 0.95 vs the input raster; output exactly symmetric; within a byte budget.

**Negative (critical)**
- **Daikonic still reconstructs to its normal stack** — the bite trigger / gate must NOT fire
  false occlusion (regression guard for the whole existing pipeline).
- a synthetic two-overlapping-disks fixture with known geometry → exact recovery.

## Scope boundaries (explicit non-goals for this spec)

- polygon / annulus completion (Olympic rings, overlapping diamonds) — extends the same
  own-boundary-fit machinery in a follow-up.
- blended / transparent overlaps — only flat **distinct-color** overlap or **pure same-color**
  occlusion; a multiply/alpha blend falls back to per-region fitting.
- N-way (>2) simultaneous overlap beyond what pairwise composition + the consistency gate
  naturally cover.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Abutment mistaken for occlusion (false positive) | bite trigger (concavity) + consistency gate rejects |
| Own arc too short to constrain a circle | require a minimum own-arc angular span; else fall back |
| Overlap is a blend, not a flat color | out of scope → per-region fitting (gate rejects the reconstruction) |
| Wrong z-order | candidate orderings disambiguated by the consistency gate |
| Performance (pairwise over many regions) | only groups with a bite-triggered region are considered |

## Acceptance summary

The feature is correct when: rasterized Mastercard idealizes to two real circles (mirror pair)
plus a lens, rendering faithfully and exactly symmetric; Daikonic and the other non-occluded
marks are unchanged; and the consistency gate guarantees any reconstruction it emits re-renders
to match the input.
