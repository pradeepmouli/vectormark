# Per-Component Fill Strategy (Colour-Step Merge) — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-25
**Area:** `gradient.py` (merge + fill decision), `pipeline.py` (`_render_body` flow); reuses `candidate.py`/`emit.py` primitives.
**Supersedes:** the whole-blob approach in `2026-06-25-gradient-fallback-design.md` (the per-component fill *primitives* from that work are retained; its whole-blob detection is replaced).
**Related:** `2026-06-06-gradient-handling-design.md` (the original gradient pipeline), `memory/gradient-segmenter-background-interaction.md`.

## Why this supersedes the whole-blob design

The first iteration unioned an entire foreground blob into one footprint and fit a
single gradient/raster to it. Validated on the corpus, that blurred sharp features
that sit on the gradient: Instagram's white camera outline and Firefox's flame
structure became smears, because the one raster conflated distinct elements and a
single bbox has low effective resolution. The fix inverts the order: **form vector
components first, then choose a fill strategy per component** — so the white outline
is simply its own component that picks `flat` and never enters the gradient/raster
decision.

## Architecture

Two steps, an incremental extension of the pre-enhancement pipeline (which already
segments into components and chooses flat-vs-gradient per component):

### 1. Unified component merge (colour-step agglomerative)

Within each gutter-component, agglomeratively merge spatially-**adjacent** regions
whose OKLab colour step is below `MERGE_TOL` into one vector component (union-find /
transitive closure over `region_adjacency`). This **generalizes** today's
`_ramp_groups` from "collinear straight ramp" to "any locally-smooth field," so
curved multi-hue arcs merge where they don't today. Regions separated by a large
colour step never merge.

The merge criterion is the whole discriminator — no median/band-count guards are
needed. Corpus measurement of adjacent-region colour steps shows a clean separating
gap between within-field steps and boundary steps:

| mark | within-field steps (→ merge) | boundary steps (→ keep separate) |
|------|------|------|
| instagram | 0.048–0.07 (gradient bands) | 0.50–0.52 (**every** large step is gradient↔`#FFFFFF` outline) |
| firefox | 0.047–0.05 (flame bands) | 0.47–0.54 (flame↔purple-sphere core) |
| gdrive | 0.093–0.12 (within-facet shading) | 0.27–0.42 (facet edges) |

The empty gap ~0.12–0.27 means any `MERGE_TOL` in **0.15–0.20** does the right thing
on all three with no special-casing. The exact value is corpus-validated.

What this yields per mark:
- **instagram**: gradient bands merge into one field component; the white outline
  (all adjacencies ≥0.50) stays a separate crisp `flat` component.
- **firefox**: flame bands merge; the sharp flame↔sphere boundary (≥0.47) splits them
  into two clean field components.
- **gdrive**: each facet's own light/dark shading merges (≤0.12), but facet *edges*
  (≥0.27) never merge → crisp facet components, each a small gradient/flat — edges
  stay sharp, and shaded facets are *more* faithful than today's flat fill.

### 2. Per-component fill strategy

**Fill-eligibility gate (restores the two guards the colour-step merge alone drops).**
Colour-step similarity is necessary but not sufficient to fit a field: real logos have
adjacent *distinct* design elements of similar colour (Daikonic's navy band shapes) that
must stay separate flat components, and individual flat regions carry mild AA variation
that a per-region fit would over-eagerly turn into a spurious gradient. A group is
eligible for a gradient/raster fill only if it is **either** a genuine merged field
(`len(group) >= _MIN_BANDS`, i.e. ≥3 bands — a posterized continuous tone) **or** a
single dominant blob (`group[0].area >= _BLOB_DOMINANCE * total_foreground_area` — the
non-posterized smooth-gradient case, e.g. a smooth disc). Ineligible groups leave their
regions in `remaining` as-is (flat). These are the existing `_MIN_BANDS = 3` and
`_BLOB_DOMINANCE = 0.85` constants — no new thresholds. Validated: under this gate
Daikonic/two-blob/gdrive stay flat (parity), while Firefox/Instagram split into per-
component fields that each fit a true parametric gradient under `_PARAM_FALLBACK_TOL`
(editable, crisp — raster is not even needed for them; it remains the safety net for
genuinely unfittable fields).

Each eligible component picks its fill independently, reusing the primitives already
built and reviewed (`fit_gradient`, `_best_parametric`, `_fit_stretch`, `RasterFill`):

1. **single unmerged region** → `flat` (its region colour) — the existing path.
2. **merged smooth field** → in order:
   a. strict `fit_gradient` (≤ `_GATE_DELTA_E`) → **linear/radial gradient** (editable, crisp).
   b. else searched `_best_parametric`, mean ΔE ≤ `_PARAM_FALLBACK_TOL` → **gradient**.
   c. else, if the field travels at least `_MIN_STOP_SPAN` → **raster** (`_fit_stretch`,
      its own tight bbox).
   d. else (degenerate, near-flat) → **flat**: the group is not consumed; its regions
      stay in `remaining` as their original regions (they share a colour, so the flat
      path renders them identically — no structural change, preserving parity).

Regions leave `remaining` only when consumed into a gradient/raster fill; everything
else (ineligible groups, degenerate-flat groups, unmerged singletons) stays as its
original regions for the flat/symmetry path. This keeps the rework structurally
minimal and is why the Daikonic symmetry/structure tests pass unchanged.

No `_SMOOTH_BAND_MIN` / `_SMOOTH_MEDIAN_TOL` / `_BLOB_DOMINANCE` guards: the merge
already decided "this is one smooth field," so the fill step only chooses the
*representation*. Removing the outline contamination up front means a field that
needed raster as a whole blob may now fit a true editable gradient per component.

## Retained from the whole-blob work (no rework)

- `RasterFill` type + `<pattern><image>` emission + `resolve_fill` (Task 1).
- Searched parametric fit + shared model builders + `_best_parametric` (Task 2).
- Adaptive `_fit_stretch` (Task 3).
- Pipeline wiring: raster model → `RasterFill`, bake via `patternTransform` (Task 5).

## Reworked / removed

- Replace the whole-blob smooth-blob path: `_fit_smooth_blob` and the
  `_dominant_blob_fraction`-gated union block in `detect_gradients`, plus the
  `_SMOOTH_BAND_MIN` / `_SMOOTH_MEDIAN_TOL` / `_PARAM_FALLBACK_TOL`-as-guard logic
  (Task 4). `_PARAM_FALLBACK_TOL` survives only as the gradient-vs-raster fill bound.
- Generalize `_ramp_groups` → a `merge_components(regions)` agglomerative colour-step
  merge that subsumes it (a straight ramp is a chain of small steps).
- `detect_gradients` becomes "merge → per-component fill," returning the same
  `(fills, remaining)` shape so `_render_body`/`build_candidates` are minimally
  touched: merged-field components become fills; unmerged singletons are `remaining`
  (flat / symmetry path, unchanged).

## Parity

The riskiest surface: replacing `_ramp_groups`/`fit_gradient`-grouping changes the
code path for **all** existing gradient handling. The generalized merge must still
group what `_ramp_groups` grouped (posterized ramps are small-step chains), and the
fill step must still emit the same linear/radial gradients for those. Acceptance
tests `test_acceptance_gradient`, `test_acceptance_smooth_gradient`, and
`test_gradient` are the parity guard; their *outputs* (one `<linearGradient>` /
`<radialGradient>` for the marks they cover) must be preserved. If unified merge
cannot preserve a specific existing-gradient output, fall back to keeping
`_ramp_groups` for the collinear case and applying the agglomerative merge only to
the residual (additive), noted as the contingency.

## Thresholds (corpus-validated, not guessed)

| name | start | role |
|------|------|------|
| `MERGE_TOL` | 0.15 | max OKLab colour step to merge two adjacent regions (the discriminator) |
| `_PARAM_FALLBACK_TOL` | 0.07 | max mean ΔE to prefer an editable parametric gradient over raster |
| `_STRETCH_TARGET`, `_STRETCH_GRID_STEPS`, `_GATE_DELTA_E`, `_MIN_STOP_SPAN` | (existing) | unchanged |

## Testing

- **Merge:** synthetic adjacent regions with small colour steps merge; a large step
  blocks the merge; transitive closure across a chain merges the whole chain.
- **Per-component fill:** a merged posterized linear ramp → `linear` gradient
  (parity); a merged 2-D field that no gradient fits → `raster`; an unmerged single
  region → flat.
- **Outline separation:** a synthetic saturated gradient adjoining a white (near-zero
  chroma) region → the gradient regions merge into one field component while the
  white region stays a separate flat component (large colour step blocks the merge).
- **Faceted art:** synthetic facets separated by large colour steps → never merge →
  one flat component per facet (crisp); within-facet shading bands (small step) merge
  into that facet.
- **Parity:** `test_acceptance_gradient`, `test_acceptance_smooth_gradient`,
  `test_gradient` keep passing — the existing-gradient marks still emit one
  linear/radial gradient.
- **Corpus:** firefox/instagram emit a gradient field component (gradient or raster)
  PLUS a crisp vector outline where applicable; gdrive stays crisp facets; no flat
  mark gains a spurious fill. Visual spot-check confirms sharp features stay sharp.

## Risks

- **Parity of the merge rework** — mitigated by the acceptance-test guard and the
  additive fallback contingency.
- **`MERGE_TOL` over-fit to three marks** — validate across the whole corpus before
  merge; require the faceted-art and outline-separation tests.
- **Merged-component geometry** — a merged field's silhouette may be a complex shape;
  `select_geometry` must fit it (it already fits arbitrary region masks). Where a
  field component is concave/holed, the existing path/holed-path fitters apply.
