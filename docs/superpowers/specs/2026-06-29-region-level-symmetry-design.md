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

Detect symmetry at the **region level**: the symmetry axis and the
straddler/pair/loner classification emerge from the regions themselves, not from
a component silhouette and not from a tuned threshold. Feed the resulting
symmetric groups to the existing (correct) reconstruction.

## Scope

- **In:** axis discovery + region classification in `symmetry.py`, and its one
  call site in `_render_body`. Vertical mirror axes only (matches today's
  `_reflect_cols`); rotational symmetry (`detect_symmetry_rotation`) is
  unchanged and out of scope.
- **Out (this change):** reconstruction (reused unchanged — already exact),
  occlusion, and surface-merge. Those keep their component pass for now. The
  new detector's output is shaped as a reusable region-relationship primitive
  so occlusion/merge can drop the component dependency in a later change with no
  rework — i.e. the component box becomes vestigial, not yet removed.

## Architecture

`symmetry.py` gains a region-level entry point that consumes the full region
list (no component box) and returns symmetric groups:

```
SymmetricGroup = (axis_x: float, straddlers: list[Region], pairs: list[tuple[Region, Region]])
detect_symmetry_groups(regions: list[Region]) -> list[SymmetricGroup]
```

### 1. Propose axes from regions (bottom-up)

For each region, propose candidate **vertical** axes:
- **Self-axis:** if the region is self-mirror-symmetric (passes the absolute
  test below about its own centroid-x, searched over a small ±window to find the
  best local axis), propose that axis-x. Weight = region area.
- **Pair-bisector:** for each pair of regions that are congruent under vertical
  reflection (the absolute test on `region_a` vs `reflect(region_b)`), propose
  the bisector x = (centroid_a.x + centroid_b.x)/2. Weight = combined area.

Pair candidate generation is pruned to plausible partners (similar area and
height, mirrored horizontal offset) so it stays near-linear, not O(n²) of full
mask comparisons.

### 2. Cluster proposals into axes

Cluster candidate axis-x values within a small pixel tolerance (e.g. 1–2px;
this is a *spatial coincidence* tolerance, not a symmetry threshold). Each
cluster whose total supporting region-area clears a minimal floor is a
**symmetry axis**. Multiple clusters → multiple axes (two symmetric icons side
by side each get their own), supported for free.

### 3. Classify every region against each surviving axis

For each axis, classify each region by the **absolute area-aware test**:
- **straddler:** `symdiff(region.mask, reflect(region.mask, axis)) ≤ k · perimeter(region.mask)` (k ≈ 1).
- **pair:** two regions whose `symdiff(a, reflect(b, axis)) ≤ k · perimeter`.
- **loner:** neither.

A region may be claimed by at most one axis (the one with the strongest support
/ best fit); ties broken deterministically (by axis-x then label) to preserve
determinism.

### The absolute decision (area-aware boundary tolerance)

A perfectly symmetric continuous shape, sampled onto a pixel mask, mismatches
its own reflection only along the boundary, by sub-pixel rounding. The criterion
encodes exactly that: the symmetric-difference area must be no more than ~1
boundary pixel thick, `symdiff_area ≤ k · perimeter`, `k ≈ 1`. This is
size-correct (a 10px glyph and a 5000px band are both held to "off by ≤1
boundary pixel") and is grounded in the sampling, not a slider. `k` is a pixel
count, justified by rasterization, not tuned to catch cases. The measured gap
(≥0.99 symmetric vs <0.65 asymmetric) means the exact `k` within a wide band is
immaterial — the decision is effectively absolute.

### 4. Output + integration

`detect_symmetry_groups` returns the groups. `_render_body`:
- calls it once on all regions (replacing per-component `detect_axis` +
  `classify_regions`),
- routes each group's straddlers/pairs to the **existing** reconstruction with
  the group's axis,
- routes loners through the normal per-region fit,
- still runs the existing per-component occlusion/merge pass (unchanged) — the
  two coexist; symmetry no longer reads the component grouping.

The detected axes feed `IdealizeReport.axes`; the per-region decisions feed
`IdealizeReport.symmetry` (same diagnostic shape as today: `(label, score,
decision)`), so the diagnostics remain a first-class regression signal.

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
- **Absolute-decision unit:** the area-aware test accepts a 1px-eroded/rounded
  reflection and rejects a genuinely asymmetric one, across small and large
  region sizes (size-correctness).
- **Corpus regression:** the byte+diagnostic baseline/diff harness above,
  including daikonic; daikonic's body shapes assert `sym-IoU = 1.0`.

## Risks / Notes

- **Pair-generation cost:** guard with area/height/offset pruning so it does not
  become O(n²) full-mask comparisons on region-dense logos.
- **Determinism:** axis clustering and region-claiming must be value-ordered
  (axis-x then label) — no dict/order dependence.
- **Component coexistence:** occlusion/merge still consume components this
  change; the new primitive must not perturb their inputs. Verify occlusion
  goldens stay byte-identical.
- **Axis precision:** a region's best self-axis is searched over a small ±window
  rather than assumed at centroid-x, so a slightly off-centroid true axis is
  still found and clusters cleanly.
