# Inflection-Guarded Cubic Bézier Fitting — Design

## Problem

Path fitting (`fit_path`) emits **quadratic** Béziers only. A quadratic has one control
point → it is always a parabola segment, so it cannot track a circular/asymmetric convex
curve well and is **segment-hungry**: a smooth boundary (e.g. the V's bottom curve) needs
many quadratics, which blows the segment budget → coarsening → the coarse chord cuts inside
the true curve → the boundary recedes from the silhouette → **gap** (the "bottom outer
boundary gap": native 2095px cluster, red↔orange at the bottom curve).

Cubics fit such curves in one segment but were **deliberately removed earlier** because a
cubic can carry an **inflection point** (S-curve) within a single segment, letting it chase
boundary noise into an unstable wiggle.

## Key fact (prototype-validated)

An **inflection-free** cubic is *not* equivalent to a quadratic — it is strictly more
expressive while remaining single-curvature (convex):

- 90° arc: one quadratic ≈ 9px max error; one cubic ≈ sub-px (≈0.5px with proper tangents) —
  both inflection-free. The cubic captures the arc; the quadratic (parabola) cannot.
- An S-curve: a naive single cubic inflects (wiggle); the **guard** detects it and splits into
  two convex cubics.

So inflection-guarding removes *only* the noise-chasing S-curve, keeping the fidelity/parsimony
win. It dissolves the cap/fraying tension at the source: convex curves fit in few segments, so
they never hit the budget and never get coarsened-cut.

## Goal

Replace quadratic path fitting with **inflection-guarded cubic** fitting: fit a single cubic
to each curve run; if it inflects in (0,1) or exceeds the error tolerance, **split and recurse**
(Schneider). Only single-curvature cubics are ever emitted — same robustness guarantee that
motivated removing cubics, now with real curve fidelity.

## Approach

New `src/vectormark/_fitcurve.py` cubic fitter (or extend the existing module):

- **`fit_cubic_bezier(points, t0, t3)`** — Schneider least-squares: endpoints `P0, P3` fixed,
  unit endpoint tangents `t0, t3` (estimated from the run, or supplied — important so the
  seam-graph can pin junction tangents later); chord-length parameterize; solve the two tangent
  magnitudes `α0, α3` by 2×2 normal equations → control points `P1 = P0 + α0·t0`,
  `P2 = P3 + α3·t3`.
- **`cubic_inflects(P0,P1,P2,P3)`** — exact: the planar cubic inflects where
  `cross(B'(t), B''(t)) = 0`, a quadratic in t; report True iff it has a root in (0,1).
- **`fit_cubics(run, max_error)`** — fit one cubic; if `cubic_inflects` OR max deviation >
  `max_error`, split the run at the point of worst deviation and recurse on each half; return
  the list of (guaranteed convex, in-tolerance) cubics.

`fit_path` (fit.py): in the corner-split loop, replace the `fit_quadratic_beziers(seg,
max_error)` call with `fit_cubics(seg, max_error)`, emitting `C` commands. The corner-split,
straight-run detection, and the segment **budget** (coarsen-to-fit) all stay — but the budget
now rarely binds because cubics need far fewer segments.

## Emit & scoring

- Emit `C{c1x} {c1y} {c2x} {c2y} {x} {y}` (cubic). `emit.py` `shape_to_path_d` and the path
  renderers already pass `d` through verbatim; the scorer's `_CMD_COST` already costs `C` at 6.
- Ellipse arcs (`_ellipse_path_d`) already use cubics — unchanged.

## Reuse by the seam-graph

`fit_cubic_bezier`/`fit_cubics` take **explicit endpoints + tangents**, so the seam-graph
(follow-up #3) reuses them directly to fit each shared edge as an open arc with endpoints
pinned to junction nodes. This is why the fitter is designed endpoint-first, not closed-loop
first.

## Determinism

Chord-length parameterization, deterministic split point (worst deviation, value-ordered ties),
no RNG. Same input → same cubics.

## Validation

- **Unit:** a sampled 90° arc fits in **1** cubic, max error < `max_error`, `cubic_inflects`
  False. A sampled S-curve yields ≥2 cubics, each `cubic_inflects` False.
- **No-inflection invariant:** every emitted cubic passes `cubic_inflects == False` (assert in
  the fitter + a property test over random convex/concave runs).
- **Budget:** `fit_path` on a smooth contour now emits far fewer commands than the quadratic
  version (assert command-count drop on a circle-ish contour) and stays within
  `MAX_PATH_SEGMENTS` without coarsening.
- **Corpus:** real-logo goldens re-derive (smoother curves, fewer commands); structure
  preserved (same shape kinds/counts); STOP+report if any shape is lost.
- **Bottom-gap:** the V-bird (conditioned) bottom interior-gap cluster shrinks materially vs
  the quadratic baseline (smoke assertion).

## Non-goals

- No change to primitive recognition (circle/ellipse/rect) or the corner-split logic.
- No reintroduction of *unguarded* cubics — the inflection guard is mandatory; an inflecting
  cubic is never emitted.
