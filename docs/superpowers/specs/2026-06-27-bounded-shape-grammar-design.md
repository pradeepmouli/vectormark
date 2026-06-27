# Bounded Shape Grammar — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-27
**Area:** `src/vectormark/selector.py`, `src/vectormark/fit.py`, `src/vectormark/score.py`, geometry-candidate generation + hole handling. Builds on the shape/fill decoupling (PR #40).

## Goal

Produce **clean silhouettes** — no frayed/serrated edges, no hole-speckle, small features (e.g. the V-bird's middle dot) intact — by treating geometric complexity as a **definition**, not an optimization. A shape *is* simple; the fitter may only emit members of a bounded simple-shape grammar. A frayed, over-segmented boundary is not a badly-approximated shape — it is **not a shape at all**, and so is never produced.

## Motivation (observed failure)

On the V-bird, the wings render with serrated edges and the interior shows white-speck holes; the middle dot is shattered. Root cause, confirmed by reading the fitter:

- `generate_geometry_candidates` already hypothesizes the *clean* shapes (a bounded polygon / rounded-trapezoid for a wing, a `circle` primitive for a dot). But it **also** offers the unbounded `fit_path` candidate, which subdivides convex quadratics until the boundary error is under `max_error` — so it faithfully traces the antialiased, quantization-jagged mask boundary. That frayed path scores **higher fidelity** than the clean shapes, and the scorer picks it. Fidelity beats simplicity today.
- The white specks are tiny **inner contours** (antialiased boundary pixels that quantize to in-between colors, form sub-`min_area` regions, get dropped → unassigned holes inside the mark) emitted as even-odd holes.
- The dot shatters because its antialiased ring quantizes away; tracing the eroded mask yields a broken core.

The complexity tradeoff is therefore mis-framed as fidelity-vs-simplicity. Under the correct framing — **a shape is constitutively simple** — the frayed path is disqualified, and the clean bounded shape wins by default. Fitting a primitive (circle) to the dot's *evidence* (its bulk) rather than tracing its eroded boundary recovers the full circle.

This is the partner change to the shape/fill decoupling (PR #40): fills are already independent of geometry, so constraining the *shape* never disturbs the gradients.

## Architecture (this spec — the core)

A region's geometry is **the best-fitting member of a bounded simple-shape grammar**, selected by fidelity among valid simple shapes only.

1. **The grammar is the candidate set.** `generate_geometry_candidates` may only emit *simple* hypotheses:
   - primitives (rect, circle, ellipse, rounded-rect, trapezoid) and the existing symmetric fits — already bounded by construction;
   - a free **polygon** bounded to ≤ `MAX_POLY_VERTICES` vertices;
   - a free **path** bounded to ≤ `MAX_PATH_SEGMENTS` convex-quadratic segments.
   The unbounded `fit_path`/`recognize_polygon` are **capped**: a fit that cannot reach the fidelity floor *within the bound* is **not produced** as a candidate. A frayed boundary (dozens of segments) is therefore unrepresentable — it can never be selected.

2. **Selection = fidelity among valid simple shapes.** The existing render-ΔE scorer (`score.py`) still ranks candidates, but the candidate set now contains only grammar members. **Parsimony becomes the membership rule (the bound), not a soft tiebreak** — simplicity is enforced before scoring, not traded against fidelity during it.

3. **Holes are shapes too.** Sub-threshold inner contours (the white specks) are not simple shapes → **dropped** (the silhouette is filled over them). A genuine, simple inner hole (a real donut) is kept — as an even-odd hole or its own shape — when it itself clears the size/simplicity bar.

4. **Fit to evidence, not to the mask boundary.** Primitive recognition (the dot → `circle`) is driven by the region's bulk/extent, so an eroded antialiased ring does not shrink or break the recovered shape.

5. **No-fit (interim, this spec).** If *no* grammar member meets the fidelity floor (a genuinely non-simple silhouette), emit the **best (loosest) bounded shape** anyway — clean but approximate — and **log** the region (strategy/ΔE) so the corpus reveals exactly which marks need decomposition. This is the seam for the fast-follow.

6. **Fills are untouched** — decoupled in PR #40.

## Out of scope (fast-follow slice, not this spec)

**Decomposition.** When no bounded grammar member fits, the region isn't one shape and should **split into simpler shapes and recurse** — via **concavity-split** (split the silhouette at its significant concave vertices into convex sub-pieces, each simple by construction; bounded, terminating, non-overlapping; genuine overlap is already handled upstream by occlusion reconstruction, so peeling's generality isn't needed). The interim no-fit behavior (#5) plus the corpus log determine which marks actually need this before we build it.

Also out of scope: **performance** (the slow occlusion `binary_erosion` and the parametric-search cost are a separate follow-up); **raster** (a fill strategy, never a shape — wiring `_fit_stretch` into `fit_fill` is the separate PR-#40 fill follow-up).

## Components / where it lands

- **`src/vectormark/fit.py`** — `fit_path` gains `MAX_PATH_SEGMENTS`; `recognize_polygon` gains `MAX_POLY_VERTICES`. Each returns `None` (no candidate) when it cannot meet the fidelity floor within the bound, instead of an over-segmented result.
- **`src/vectormark/selector.py`** — `generate_geometry_candidates` only appends bounded candidates; drops the unbounded path/polygon; threads the no-fit log.
- **`src/vectormark/score.py`** — parsimony as membership (bound) rather than soft cost; the render-ΔE fidelity floor used to admit/reject a bounded fit.
- **Hole handling** (contour/holed-path emission in `selector.py`/`fit.py`) — drop sub-threshold inner contours.

## Tunable constants (calibrated against the corpus, not assumed)

- `MAX_PATH_SEGMENTS` ≈ 12 (convex quadratics).
- `MAX_POLY_VERTICES` ≈ 10.
- Fidelity floor: a render-ΔE threshold (region-mean) above which a bounded fit is rejected / the no-fit path fires.
- Hole-drop threshold: minimum inner-contour area (relative to the mark, matching the existing resolution-independent `min_region_fraction` convention) below which an inner contour is filled rather than emitted.

## Testing

- **Acceptance (V-bird):** wings emit as clean bounded polygons/trapezoids and dots as clean circles (not frayed paths); no white-speck holes; no serration; the middle dot is a single full-size circle. Verified visually + by element/segment counts (no path exceeds the segment bound).
- **Corpus regression:** the bound must NOT degrade currently-good outputs — genuinely-curvy-but-simple logos must still fit within the bound (calibrate `MAX_PATH_SEGMENTS` so they survive); flats unchanged. Compare before/after element + segment counts and render-ΔE; flag any mark that newly hits the no-fit path (a decomposition candidate).
- **Unit:** `fit_path` returns `None` past the segment bound when error can't be met; `recognize_polygon` respects the vertex cap; a sub-threshold inner contour is dropped; a circle is recovered from an eroded-ring synthetic dot.

## Risks

- **Bound too low** → over-rejects a legitimately curvy simple logo → it hits the no-fit interim (loose fit) or degrades. Mitigation: calibrate `MAX_PATH_SEGMENTS`/`MAX_POLY_VERTICES` against the corpus; the bound must be generous enough to keep good outputs while disqualifying fraying (dozens of segments).
- **Interim no-fit approximation** → an intricate mark that needs decomposition will look clean-but-loose until the fast-follow lands; mitigated by logging so it's visible, not silent.
- **Hole-drop too aggressive** → a genuine small inner shape (a real hole/counter) gets filled. Mitigation: the threshold mirrors `min_region_fraction`; genuine counters in logos are typically above it.
