# Region-Level Symmetry Detection — Design

## Problem

Symmetry is currently detected at the **component** level: `_render_body` groups
regions into gutter-separated components (`decompose_components`), unions each
component's region masks into a silhouette, and runs `detect_axis` on that
silhouette. A region-set is judged symmetric by a **tunable IoU threshold**
(`SYM_TOL`/`STRADDLE_MIN_IOU`/`pair_iou`).

This is wrong on two foundational counts:

1. **Wrong unit.** The component silhouette mixes regions that don't belong to
   the same symmetric figure. On the `daikonic` fixture the radish icon and the
   "Daikonic" wordmark fall into one component (their gap, 12px, is the same
   scale as the gaps *between* the radish's own colour bands, 13–14px, so
   gutter splitting cannot separate them). The combined silhouette scores
   mirror-IoU **0.64**, so `detect_axis` returns `None` and the radish's real
   symmetry is never found. Measured in isolation, the radish alone scores
   **0.993** and its axis is found at x=323 — the symmetry is real and
   detectable; it is only *averaged away* by the wrong unit.

2. **Threshold, not absolute.** Symmetry is an absolute property — a figure
   either is mirror-symmetric or it is not. The measured data shows no
   continuum: genuinely-symmetric designed shapes score **≥0.99**, everything
   asymmetric scores **<0.65**. A tunable acceptance threshold both implies a
   gradient that does not exist and invites "fixing" a miss by sliding the dial
   instead of fixing the real fault.

A consequence observed during investigation: when detection *does* fire, the
existing reconstruction already emits **exactly** symmetric geometry
(`sym-IoU = 1.0000`, mirror-one-half). The 0.99-not-1.0 we saw on full daikonic
was purely the faithful per-contour fallback after detection failed to fire.
**Reconstruction is correct; only detection is broken.**

## Goal

Detect symmetry at the **region level**: the symmetry axes and the
straddler/pair/loner classification emerge from the regions themselves, not from
a component silhouette and not from a tuned threshold. A figure may have
**multiple mirror axes at any orientation** (a square has up to four); detect
them all. Feed the resulting symmetric groups to the existing (correct)
reconstruction.

## Scope

- **In:** axis discovery + region classification in `symmetry.py`, and its one
  call site in `_render_body`.
  - **Multiple axes, any orientation** are *detected* and reported in the
    diagnostics. Detection reuses the existing resampling-free angle-generic
    reflection primitive `_axis_mismatch` (which already reflects point
    coordinates across a line at arbitrary angle θ through a center and scores
    the mismatch via the background distance transform) — so going beyond
    vertical is wiring, not new geometry.
  - **Reconstruction is enforced about the PRIMARY axis only** (the
    strongest-supported / largest-area axis) this pass, reusing the existing
    single-mirror machinery — which already handles a non-vertical axis by
    rectifying the frame so the axis becomes vertical, then mirroring. So a
    non-vertical primary axis is supported with no new reconstruction code.
- **Out (this change):**
  - **Full dihedral reconstruction** (folding a figure into its fundamental
    domain and replicating across *every* detected axis) is deferred to a
    follow-up. This pass detects all axes but only makes the output exactly
    symmetric about the primary one.
  - Reconstruction internals (reused unchanged — already exact for a single
    axis), occlusion, and surface-merge keep their component pass for now. The
    new detector's output is shaped as a reusable region-relationship primitive
    so occlusion/merge can drop the component dependency in a later change with
    no rework — i.e. the component box becomes vestigial, not yet removed.
  - Rotational symmetry (`detect_symmetry_rotation`) and the rectified path are
    unchanged.

## Architecture

`symmetry.py` gains a region-level entry point that consumes the full region
list (no component box) and returns symmetric groups. An axis is a line
`(theta, cx, cy)` (angle + a point it passes through), not just an x:

```
Axis2D       = (theta: float, cx: float, cy: float)         # mirror line at any orientation
SymmetricGroup = (axes: list[Axis2D],                       # all mirror axes of this figure, primary first
                  straddlers: list[Region],
                  pairs: list[tuple[Region, Region]])
detect_symmetry_groups(regions: list[Region]) -> list[SymmetricGroup]
```

`axes[0]` is the primary (largest supporting area); reconstruction uses it,
diagnostics report all of them.

### 1. Propose axes from regions (bottom-up, any orientation)

For each region, propose candidate axes at **any orientation**:
- **Self-axis:** find each axis through the region's centroid about which the
  region is self-mirror-symmetric — sweep θ over [0, π) at a coarse step,
  score each with the absolute test below (resampling-free, via `_axis_mismatch`
  reflecting point coordinates), and refine the peaks. Each accepted θ is a
  proposal. Weight = region area.
- **Pair-bisector:** for each pair of regions congruent under reflection, the
  mirror line is the **perpendicular bisector** of the segment joining their
  centroids; propose `(theta, midpoint)`. Weight = combined area.

Pair candidate generation is pruned to plausible partners (similar area,
similar second moments / shape, centroid separation consistent with a shared
figure) so it stays near-linear, not O(n²) of full mask comparisons.

### 2. Cluster proposals into axes

Cluster candidate axes in `(theta, signed-distance-from-origin)` space within a
small tolerance (≈1–2° in angle, ≈1–2px in offset; a *spatial coincidence*
tolerance, not a symmetry threshold). Each cluster whose total supporting
region-area clears a minimal floor is a **symmetry axis**. A figure that yields
several clusters (e.g. vertical + horizontal) keeps them **all** — multiple
axes; multiple independent figures yield multiple groups.

### 3. Classify every region against each surviving axis

For each axis, classify each region by the **absolute test** (below):
- **straddler:** the region reflected across the axis lands on itself within the
  rasterization band.
- **pair:** two regions where one reflected across the axis lands on the other
  within the band.
- **loner:** neither.

A region may be claimed by at most one figure/axis-group (the one with the
strongest support / best fit); ties broken deterministically (by axis θ, then
offset, then label).

### The absolute decision (area-aware boundary tolerance, any orientation)

A perfectly symmetric continuous shape, sampled onto a pixel mask, mismatches
its own reflection only along the boundary, by sub-pixel rounding. The criterion
encodes exactly that: the reflected foreground that lands **off** the shape must
be no more than a ~1px-thick boundary band — `off_area ≤ k · perimeter`,
`k ≈ 1`. It is computed the **resampling-free** way `_axis_mismatch` already
uses: reflect the foreground *coordinates* across the line and look each up in
the background distance transform (no whole-raster rotation, no staircasing),
counting those farther than ~1px off-shape. This:
- is size-correct (a 10px glyph and a 5000px band are both held to "off by ≤1
  boundary pixel"), grounded in the sampling, not a slider;
- works at **any orientation** for free, since `_axis_mismatch` is angle-generic;
- needs no reverse test — reflection preserves area, so "reflected ⊆ shape
  within the band" already implies bilateral symmetry about the line.

`k` is a pixel count justified by rasterization, not tuned to catch cases. The
measured gap (≥0.99 symmetric vs <0.65 asymmetric) means the exact `k` within a
wide band is immaterial — the decision is effectively absolute.

### 4. Output + integration

`detect_symmetry_groups` returns the groups. `_render_body`:
- calls it once on all regions (replacing per-component `detect_axis` +
  `classify_regions`),
- routes each group's straddlers/pairs to the **existing** reconstruction using
  the group's **primary axis** (`axes[0]`); a non-vertical primary axis goes
  through the existing rectify-then-mirror path, so no new reconstruction code,
- routes loners through the normal per-region fit,
- still runs the existing per-component occlusion/merge pass (unchanged) — the
  two coexist; symmetry no longer reads the component grouping.

All detected axes (not just the primary) feed `IdealizeReport.axes`; the
per-region decisions feed `IdealizeReport.symmetry` (same diagnostic shape as
today: `(label, score, decision)`), so the diagnostics remain a first-class
regression signal and the secondary axes are visible even though only the
primary is reconstructed this pass.

## Validation / Regression Gate

The rework changes symmetry decisions, so correctness is verified by a
**before/after diff** over the full corpus, not by trusting the new output:

1. **Corpus:** the existing 20 logos **plus `daikonic`** (promoted from fixture
   into the corpus set).
2. **Baseline capture (before any code change):** for every corpus logo, record
   (a) the exact **SVG bytes** and (b) the **symmetry diagnostics** (detected
   axes + per-region straddler/pair/loner decisions) from current `master`/base.
3. **After the rework:** re-run and diff both signals per logo.
   - **Byte-identical AND diagnostic-identical → pass silently** (no review).
   - **Changed → "drops out"** for review.
4. **Visual review only the dropped-out set**, and by a human reviewer — Haiku
   has proven unreliable at logo-fidelity judgement and is not a gate here.
5. **Adjudication is case-by-case.** Each changed logo is judged on its own:
   `daikonic` is expected to change (gains exact symmetry — the goal); any other
   change is examined and accepted or fixed individually. Non-negotiables: no
   currently-symmetric mark loses symmetry, and no false symmetry is introduced
   (the known false positives `icloud` ~0.90 and `telegram` must stay rejected —
   their regions do not pass the absolute area-aware test).

## Testing

- **Unit (`symmetry.py`):** synthetic region sets — a single self-symmetric
  region (straddler); two mirror regions (pair); a self-symmetric region + an
  off-axis asymmetric region (axis still found, asymmetric one is a loner,
  i.e. the daikonic radish+text shape in miniature); two independent symmetric
  groups at different axes (both found); a near-symmetric-but-not region at the
  icloud score (rejected by the absolute test). Determinism: identical output
  across repeated runs.
- **Multi-axis / orientation unit:** a square region yields multiple axes
  (vertical + horizontal at minimum) with the primary chosen deterministically;
  a region symmetric about a **non-vertical** axis (e.g. 45°) is detected and
  its primary axis reconstructs correctly through the rectify path; a
  horizontally-mirrored region pair is detected.
- **Absolute-decision unit:** the area-aware test accepts a 1px-eroded/rounded
  reflection and rejects a genuinely asymmetric one, across small and large
  region sizes (size-correctness).
- **Corpus regression:** the byte+diagnostic baseline/diff harness above,
  including daikonic; daikonic's body shapes assert `sym-IoU = 1.0`.

## Risks / Notes

- **Pair-generation cost:** guard with area/shape/offset pruning so it does not
  become O(n²) full-mask comparisons on region-dense logos. The θ-sweep for
  self-axes is coarse-then-refine to bound the angle search.
- **Determinism:** axis clustering, primary-axis selection, and region-claiming
  must be value-ordered (θ, then offset, then label) — no dict/order dependence.
  Primary-axis tie-break (equal supporting area) is resolved by this same order.
- **Component coexistence:** occlusion/merge still consume components this
  change; the new primitive must not perturb their inputs. Verify occlusion
  goldens stay byte-identical.
- **Axis precision:** a region's best self-axis is searched over a small ±window
  rather than assumed at centroid-x, so a slightly off-centroid true axis is
  still found and clusters cleanly.
