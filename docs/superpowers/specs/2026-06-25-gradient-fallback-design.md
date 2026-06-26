# Layered Gradient Fallback for Smooth Blobs — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-25
**Area:** `gradient.py`, `candidate.py`, `emit.py`
**Related:** `docs/superpowers/specs/2026-06-06-gradient-handling-design.md` (the original gradient pipeline this extends), `memory/gradient-segmenter-background-interaction.md`, `memory/symmetry-acceptance-gates.md` (the "loosening a gate → artifacts" discipline applied here).

## Problem

Smooth multi-hue logo marks (Firefox flame, Instagram camera) are rendered as
many stair-stepped flat bands (29 / 37 `holed_path` shards) instead of one smooth
fill. `detect_gradients` never accepts a gradient for them.

### Root cause (measured, not assumed)

For a single dominant blob the smooth-gradient path runs `fit_gradient` on the
whole silhouette against the strict consistency gate `_GATE_DELTA_E = 0.05` (mean
OKLab ΔE). The fit is **correctly** rejected — these fields are not expressible as
one linear or radial gradient. An exhaustive global centre/angle search confirms
the floor:

| mark | best radial mean ΔE | best linear mean ΔE | strict gate |
|------|------|------|------|
| firefox | 0.109 | 0.116 | 0.05 |
| instagram | 0.129 | 0.143 | 0.05 |
| gdrive | 0.113 | 0.126 | 0.05 |

So **fit quality is not the lever** — even the globally-optimal single gradient is
2–3× over the gate. Firefox is an angular/swirl field; Instagram is a 2-D
corner-anchored hue sweep. Neither is a `linearGradient`/`radialGradient`. The
fragmentation is the *fallback* when no gradient is accepted.

### The discriminator problem

`gdrive` is flat faceted art (6 regions, not a gradient) yet a radial "fits" it at
mean ΔE 0.113 — the *same range* as the real gradients. So mean ΔE alone cannot be
relaxed: it would swallow gdrive's crisp facets into a bogus blurred fill. Two
structural signals separate them cleanly (measured):

| mark | bands | **median per-pixel ΔE** | mean ΔE |
|------|------|------|------|
| firefox | 29 | **0.039** | 0.109 |
| instagram | 37 | **0.049** | 0.129 |
| gdrive | 6 | **0.093** | 0.113 |

A real gradient is a posterized continuous tone: **many** quantized bands, and the
**bulk** of its pixels follow *some* smooth model tightly (low median) — the high
mean comes from a small badly-fit minority (the swirl core). Faceted art has few
regions and **no** smooth model fits even its median pixel.

### The stretch-fill insight (measured)

A gradient is a 1-D strip stretched along an axis. The 2-D generalization:
downsample the footprint to NxN and let the SVG renderer stretch it back with
bilinear interpolation. This captures any smooth field and beats the parametric
floor:

| mark | 8×8 | 16×16 | 32×32 | parametric floor |
|------|------|------|------|------|
| firefox | 0.113 | 0.067 | **0.038** | 0.109 |
| instagram | 0.130 | 0.087 | **0.046** | 0.129 |
| gdrive | 0.100 | 0.054 | **0.029** | 0.113 |

At 32×32 (~2.6 KB base64) every mark reaches ΔE 0.03–0.05. **gdrive reaches it
too** — by reproducing its facets *blurred*. ΔE cannot see the lost crispness
(it is a minority of edge pixels; gdrive's p90 = 0.110). So the stretch-fill must
be gated by the structural smoothness guard, never by its own ΔE.

## Goals

- Firefox / Instagram emit one smooth fill instead of 29 / 37 flat shards.
- Genuine clean gradients still emit **editable** parametric gradients.
- Faceted flat art (gdrive) stays crisp flat facets — never blurred.
- Full parity for every mark that idealizes acceptably today.

## Non-goals (YAGNI)

- Focal-point / elliptical radial gradients.
- Conic / mesh gradients (not natively SVG-expressible; covered by stretch-fill).
- Making the strict band-merge path looser (it is correctness-sensitive; untouched).
- A scored raster candidate competing in the render-ΔE scorer (it would win every
  region at ΔE ≈ 0 and turn the tool into a raster-clipper).

## Design

### Scope of change

Only the **smooth-blob fallback branch** of `detect_gradients` changes — the
`if _dominant_blob_fraction(sil) >= _BLOB_DOMINANCE` block. The band-grouping path
above it, `fit_gradient`'s strict 0.05 acceptance, and every other code path are
unchanged. Single-component / band-groupable marks therefore produce identical
output.

### Decision ladder

For a single smooth dominant blob the band-merge path did not consume:

```
a. Proper parametric fit (searched linear + radial) → best model
   compute mean and median per-pixel ΔE of that model over the footprint

b. Smoothness guard — is this a gradient-like field at all?
     band_count >= BAND_MIN  AND  median_per_pixel_dE <= MEDIAN_TOL
   fail → leave as flat bands   (gdrive: 6 bands, median 0.093 → faceted, untouched)

c. mean_dE <= PARAM_TOL → emit PARAMETRIC gradient (editable)   [genuine gradients]

d. else → adaptive STRETCH-FILL                                  [2-D fields]
     downsample footprint bbox to NxN, grow N over GRID_STEPS until
     mean_dE <= STRETCH_TARGET; if the cap is reached first, emit the best N
     (still beats fragmentation). Emit as a RasterFill (<pattern><image>).
```

The median guard gates **both** the parametric (c) and stretch-fill (d) tiers, so
faceted art can never reach the stretch-fill even though its mean ΔE would pass.

### Proper parametric fit

Replace the centre/axis *heuristic* used inside the fallback with a deterministic
coarse→refine search:

- **Radial:** evaluate candidate centres on a grid spanning the footprint bbox
  extended ±50% on each side, then refine locally around the best; `r = max radius
  from that centre`; keep the centre with the lowest mean ΔE.
- **Linear:** evaluate axis angles over a fixed step (e.g. 5°), refine locally;
  project, fit stops, keep the lowest mean ΔE.

Determinism: fixed grids/steps, no randomness, stable input order. The search is
used only on the fallback blob (one per component), so cost is bounded.

The strict band-merge path keeps calling the existing `fit_gradient` unchanged.
The searched fit is a separate fallback-only routine (it must not alter the
strict path's accept/reject behavior).

### Types and SVG emission

- `candidate.py`: add

  ```python
  @dataclass
  class RasterFill:
      """A bilinear-stretched raster fill: a small NxN PNG stretched across
      `geometry` = {x, y, w, h} (the footprint bbox) and clipped by the path it
      fills. Used for smooth 2-D colour fields no parametric gradient expresses."""
      geometry: dict
      png_b64: str
  ```

  and extend `Fill = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill`.

- `emit.py`: add `pattern_image_def(elem_id, x, y, w, h, png_b64)` emitting

  ```xml
  <pattern id="ID" patternUnits="userSpaceOnUse" x="X" y="Y" width="W" height="H">
    <image href="data:image/png;base64,..." x="X" y="Y" width="W" height="H"
           preserveAspectRatio="none"/>
  </pattern>
  ```

  Extend `resolve_fill` so a `RasterFill` registers this def (id `g{len(defs)}`,
  minted before append) and returns `url(#id)` — identical plumbing to gradients.
  The path being filled is the clip; the pattern spans exactly the bbox so it does
  not tile within the region. `userSpaceOnUse` + absolute coords means it survives
  `--flatten` and bakes through the existing `geometry`-override path in
  `resolve_fill`.

### Thresholds (corpus-validated, not guessed)

Starting values, to be tuned against the full corpus gallery so no real mark
regresses (same discipline as the symmetry-gate fix):

| name | start | role |
|------|------|------|
| `BAND_MIN` | 10 | min quantized bands in the blob — `len(leftover)`, the regions forming the fallback silhouette (posterized-continuous signal) |
| `MEDIAN_TOL` | 0.05 | max median per-pixel ΔE of the best parametric fit (field is smooth) |
| `PARAM_TOL` | 0.07 | max mean ΔE to prefer the editable parametric gradient |
| `STRETCH_TARGET` | 0.05 | stretch-fill grid grows until mean ΔE ≤ this |
| `GRID_STEPS` | (8,16,24,32,48) | candidate NxN grid sizes; last is the cap |

All live as named module constants in `gradient.py` beside the existing
`_GATE_DELTA_E` / `_BLOB_DOMINANCE`, each with a one-line rationale comment.

## Testing

- **Parametric tier:** synthetic smooth linear and radial gradients, posterized to
  N≥`BAND_MIN` bands → accepted as `LinearGradientFill` / `RadialGradientFill`,
  re-render within `PARAM_TOL`.
- **Stretch-fill tier:** a synthetic separable 2-D field (e.g. horizontal hue ×
  vertical luminance, or diagonal × radial) that no single gradient fits under
  `PARAM_TOL` → emitted as `RasterFill`, re-renders within `STRETCH_TARGET`; the
  embedded grid is the smallest `GRID_STEPS` entry that meets the target.
- **Faceted-art guard:** synthetic flat faceted art (a few large flat regions,
  median per-pixel ΔE of best fit > `MEDIAN_TOL`) → stays flat regions, emits **no**
  `RasterFill` and no gradient (the blur regression guard).
- **Emission:** `resolve_fill(RasterFill(...))` registers exactly one `<pattern>`
  def and returns `url(#g0)`; the `<image>` carries `preserveAspectRatio="none"`
  and a valid `data:image/png;base64,` href.
- **Parity:** existing `test_gradient*` suites pass unchanged; a band-groupable
  posterized ramp still takes the strict band-merge path (no behavior change).

## Risks

- **Threshold over-fit to three marks.** Mitigation: validate every constant
  against the whole corpus before merge; require the faceted-art guard test.
- **Renderer image-smoothing differences.** `preserveAspectRatio="none"` +
  default `image-rendering` is bilinear in all mainstream renderers (resvg,
  browsers); the scorer uses resvg, matching delivery.
- **`<use>`-mirrored raster regions.** Rare (gradient regions seldom pair). A
  userSpaceOnUse pattern mirrors with the shape under a `<use transform>`, which is
  correct. Not specially handled; noted for the implementer.
