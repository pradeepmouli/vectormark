# Antialiasing-aware sub-pixel boundary extraction — design

**Status:** design (seeds a writing-plans plan)
**Branch:** `feat/aa-contours`
**Date:** 2026-06-27
**Related:** `2026-06-27-bounded-shape-grammar-design.md` (the tolerance partner), memory `boundary-noise-antialiasing-roadmap`, `2026-06-07-perceptual-palette-design.md` (palette this consumes)

---

## Goal

Extract each region's boundary from the **soft, pre-quantization antialiased signal** instead of the hard binary mask, so contours are smooth **by construction** — no quantization staircase — while guaranteeing that **adjacent regions share an identical sub-pixel seam (no gaps, no overlaps)**.

The no-gap shared-seam guarantee is the **central, non-negotiable requirement** and the spine of this design. Everything else (smoothness, sub-pixel accuracy) is layered on top of a construction that makes a single consistent partition of the image the primitive, not a bag of independently-traced per-region contours.

## Motivation — the staircase is manufactured, not inherent

Source raster logos have **smooth** edges because boundary pixels are **antialiased**: a 1–2px band of color blends between the two sides. vectormark destroys that smoothness in three deterministic steps:

1. **`color.quantize` (color.py:137, argmin at 143–145)** computes each pixel's OKLab distance to every palette color and takes `argmin` — every antialiased blend pixel is hard-assigned to one label. The continuous edge collapses to a per-pixel binary decision.
2. **`segment.segment` (segment.py:22, mask at 31)** builds a bilevel mask per color via `np.all(quantized == color, axis=2)`.
3. **`contour.region_contours` / `outer_contour` (contour.py:21–38)** run `find_contours(mask.astype(float), 0.5)` on that `{0,1}` field. On a binary field every 0.5-crossing snaps to a half-pixel grid line → a **staircase**. Marching-squares faithfully traces noise that step 1 invented.

The boundary noise is **quantization staircasing of smooth AA edges**. `find_contours` already does linear sub-pixel interpolation between samples — feed it a *continuous* field and it yields a smooth contour for free. So the fix is upstream of marching squares: give it a soft field.

This is the principled partner to the bounded-shape-grammar work, which **tolerates** the staircase with robust RMS recognition. If boundaries are smooth at the source, that tolerance becomes headroom rather than a crutch (see *Interaction*).

## Approaches surveyed

### A. Soft per-region membership → `find_contours` at 0.5
Replace the binary mask with a continuous per-pixel membership `L_k(x,y) ∈ [0,1]` (region `k`'s coverage of the pixel), traced at the 0.5 level.
- **Real?** Yes — this is just `find_contours` on a different field; skimage gives the sub-pixel interpolation. Nothing exotic.
- **Tradeoff:** smooth by construction, but if the membership is a generic similarity (e.g. softmax of −ΔE) the 0.5 level only *approximately* matches the true geometric edge, and two regions traced **independently** can disagree at the seam → **gap/overlap**. Independent per-region tracing cannot guarantee no-gap.

### B. Alpha-unmix / coverage-from-blend
An AA boundary pixel is a linear blend `V = α·C_A + (1−α)·C_B`. Knowing the two colors, recover coverage exactly:
`α = clip( (V−C_B)·(C_A−C_B) / |C_A−C_B|² , 0, 1 )`.
- **Real?** Yes — this is the standard alpha/coverage inversion (matting, sub-pixel edge localization). Exact for a clean two-color edge.
- **Tradeoff:** geometrically **exact**, but needs the correct `(C_A, C_B)` pair per boundary segment and is **ill-posed at junctions** (3+ colors meet → underdetermined).

### C. Bespoke de-antialiasing / depixelizing tracers (literature)
Kopf–Lischinski depixelizing-pixel-art, Potrace-style optimal polygons on coverage, sub-pixel Canny, curve-stitching vectorizers.
- **Real?** The algorithms exist, but they **reimplement** what `find_contours` on a soft field gives us, add heavy machinery (similarity graphs, spline solvers), and most are tuned for upscaling pixel art, not for our "we already have the pre-quant RGB + palette" situation. **Rejected as primary** — we'd rebuild skimage's interpolation. Worth citing as prior art only.

### The decisive insight: signed margin makes seams shared by construction
Neither A nor B alone guarantees no-gap. The fix is to stop thinking "per-region contour" and define a **signed margin field** per region:

> `φ_A(x,y) = L_A(x,y) − max_{k≠A} L_k(x,y)`  — positive inside A, negative outside, **zero on A's boundary**.

A's boundary is the **zero-level set** of `φ_A`. The guarantee:

- **Along any two-color A|B seam**, the runner-up for A is B and the runner-up for B is A, so
  `φ_A = L_A − L_B` and `φ_B = L_B − L_A = −φ_A` **exactly** (bitwise negation in float).
- `find_contours` localizes a zero-crossing on a grid edge at `t = φ(p)/(φ(p)−φ(q))`. For `φ_B = −φ_A` this is `−φ_A(p)/(−φ_A(p)+φ_A(q)) = φ_A(p)/(φ_A(p)−φ_A(q))` — **the same `t`**. The marching-squares sign pattern is mirrored, so the same cells are active. ⇒ The shared sub-arc of A's and B's contours is **point-for-point identical**. No gap, no overlap — *proven numerically*, not just approximately.

This converts the user's hard constraint into an arithmetic identity. To keep it exact in code we **compute each seam once and assign `φ` to one side, `−φ` to the other** (don't recompute the neighbor's field independently — see seam below). Alpha-unmix (B) then refines *position*: at a clean seam we make `φ_A = 2α − 1`, whose zero sits at `α = 0.5` = the true sub-pixel edge. Because `φ_B := −φ_A`, the refinement moves **both** sides together — the seam stays shared while becoming exact.

## Recommended approach

**A hybrid signed-margin partition:** a global soft label field is the foundation that guarantees a single gap-free partition; alpha-unmix refines each seam's sub-pixel position without ever splitting it into two curves.

### 1. Build the global soft label field (foundation → no-gap)
Computed once, where the pre-quant RGB and palette are both in scope: `_segment_image` (pipeline.py:62–63, `arr` + `palette`). Note the **background color is a palette label too** (the mark-vs-background edge is the most common seam) even though `segment` excludes it as a Region (segment.py:29).

Per pixel, produce memberships `L_k`, partition-of-unity (`Σ_k L_k = 1`), by band:

- **Interior (hard-label core):** pixels >~1.5px from any label change keep **one-hot** membership (`L = 1` for their color). *Anchoring the interior is what prevents thin-feature collapse* (a 1px AA stroke whose blend never reaches 0.5 would otherwise vanish).
- **Two-color band (the common case):** identify the two dominant palette colors `C_A, C_B` in the local neighborhood; alpha-unmix `α` (formula above); set `L_A = α, L_B = 1−α`, others 0. This is the geometric-exact path.
- **Junction band (≥3 colors locally, unmix ill-posed):** fall back to normalized inverse-ΔE membership (value-tiebroken for determinism, mirroring `extract_palette`'s ordering). Smooth and partition-of-unity, position approximate — acceptable because junctions are corners anyway.

### 2. Derive per-region signed margin, attach to Region
`φ_k = L_k − max_{j≠k} L_j`. Map to a coverage-style field `cov_k = (φ_k + 1) / 2 ∈ [0,1]` so the boundary sits at **0.5** — this keeps `contour.py`'s existing `find_contours(..., 0.5)` call **unchanged**. Attach `cov_k` to the region.

### 3. The pipeline seam / interface (small, stable, backward-compatible)
The bool `mask` **stays** on `Region` and every existing consumer keeps using it untouched — symmetry IoU (symmetry.py:146–162), occlusion (occlusion.py), score (score.py:112), surface_merge (surface_merge.py). **Only contour extraction reads the soft field.**

```python
# types.py — Region gains one optional field (default None = today's behavior)
@dataclass
class Region:
    label: int
    mask: np.ndarray                 # bool (H,W) — UNCHANGED, all other consumers use this
    color_hex: str
    coverage: np.ndarray | None = None   # float (H,W), region's α field, boundary at 0.5

# contour.py — one source switch, level stays 0.5
def region_contours(mask, *, coverage=None):
    field = coverage if coverage is not None else mask.astype(float)
    padded = np.pad(field, 1)                       # coverage already 0 outside the region
    contours = find_contours(padded, 0.5)
    ...                                              # unchanged from here

def outer_contour(mask, *, coverage=None): ...      # same switch
```

Callers (`selector.py:56`, `occlusion.py:45`, `contour.region_corner_radius`) pass `region.coverage` when present; with `coverage=None` the function is byte-for-byte today's behavior. A new module (`softlabel.py`, or extend `color.py`) holds `soft_label_field(rgb, palette, hard_labels)` and `region_margin(field, k)`. `_segment_image` computes the field once and attaches `coverage` to each Region — nothing downstream of segmentation changes shape.

### 4. Seam-once discipline (keeps the guarantee exact)
Because `φ_B = −φ_A` only holds when both sides' fields are derived from the **same** `L`, compute the global `L` once and derive every `cov_k` from it. Do **not** recompute a region's field from only its own pixels. Strategy 1 (field-based, above) inherits the negation-exactness proof and is the recommendation. A heavier **Strategy 2 — explicit planar seam graph** (junction nodes, each seam an edge shared by reference) is the documented fallback **only if** the junction sliver test (below) shows slivers above tolerance; do not build it preemptively.

## Interaction with bounded-grammar / robust fitting

They **compose, they don't conflict** — this removes noise at the source; the grammar tolerates whatever residual remains.

- A bilevel staircase has ±0.5px quantization → contour RMS to the true line/arc ≈ **0.3–0.5px**. A soft-field contour localizes the edge to ≈ **0.05–0.15px**.
- The bounded grammar currently uses a **loose** robust-RMS gate to recognize a circle/rect/rounded-rect through the staircase. With smooth contours that gate has headroom to **tighten back toward exact**, so primitives win more often and the generic-`fit_path` fallback fires less.
- **Sequencing:** ship AA-contours first behind its default-on attachment; then, as a *separate* follow-up, revisit the bounded-grammar RMS threshold downward. Keep them decoupled — don't couple this PR to a grammar-threshold change.

## Determinism

- All numpy-vectorized; no RNG, no thread-order dependence. `find_contours` is deterministic.
- Alpha-unmix is pure arithmetic. The junction fallback's color selection and any ties are **value-ordered** (lexsort), matching `extract_palette` (color.py:85). Two runs ⇒ bit-identical contours (asserted in tests).

## Testing

1. **Gap-free shared seam (the spine).** Synthetic two-color image, vertical AA ramp at the seam. Extract both regions. Assert the shared sub-arcs are **point-identical** (≤1e-9). Supersample-rasterize both filled contours: every seam-band pixel covered by **exactly one** region — zero uncovered, zero double-covered.
2. **Smoothness.** AA-rendered circle + AA diagonal edge: contour RMS to the true primitive **< 0.15px**, and strictly **better than** today's `find_contours(mask, 0.5)` baseline on the same input.
3. **Sub-pixel accuracy.** AA edge placed at a fractional position (e.g. x = 10.3): extracted boundary localizes within **~0.1px**.
4. **Junction / triple point.** Three colors meeting (Mercedes-star / pizza split): all three contours pass within ε of a common point; **no sliver gap or overlap > tolerance** at the triple point. (If this fails → escalate to Strategy 2 junction snapping.)
5. **Thin feature.** 1px AA stroke survives (interior one-hot anchoring) and does not collapse.
6. **Determinism.** Run twice ⇒ bit-identical contours.
7. **Backward-compat.** `coverage=None` ⇒ `region_contours` output equals the pre-change bilevel result exactly.
8. **Real fixture (V-bird).** Blue-wing / orange-wing shared seam is gap-free and visibly smoother than baseline; mark-vs-background silhouette smooth.

## Risks

- **Triple-point slivers (primary).** At a junction cell the three zero-level curves may route through slightly different sub-cell points. Mitigation: snap all contours crossing a detected junction cell (≥3 labels within ε of the max) to the shared centroid; or accept isolated sub-pixel slivers if test 4 shows them below tolerance. Build the snap only if needed.
- **Thin features.** Covered by interior one-hot anchoring; without it a fully-AA hairline could never reach membership 0.5. Tested (test 5).
- **Background/foreground.** Background must be included as a competitor label in `L` (it isn't a Region but it is the other side of the most common seam). `_background_color` (segment.py:15) identifies it post-quant; the field must list it among the palette labels.
- **Gradients.** A gradient surface has no palette-label change *inside* it, so no spurious internal seam is created — coverage extraction runs at **region boundaries only**; `surface_merge` continues to own gradient seams downstream. Do not over-smooth or invent edges within a gradient body.
- **Palette gaps.** A pixel whose true color was dropped by the palette seed-floor rule (color.py:98–101) unmixes imperfectly; low-frequency by construction and falls back to membership. Safe direction (a miss, not a false seam).
- **Holes / counters.** `φ_A` is negative inside a hole, so `find_contours` returns outer + hole contours naturally; existing area-sort (contour.py:27) is unaffected.

## Out of scope

- Changing quantization or the palette (color.py stays; we *consume* its palette + the pre-quant RGB).
- Changing the fitters / scorer, or the bounded-grammar RMS threshold (separate follow-up).
- Changing the bool-`mask` consumers (symmetry, occlusion, score, surface_merge) — they keep using `mask`.
- Gradient seam logic (`surface_merge`) — unchanged.
- The full planar seam-graph (Strategy 2) unless the junction test forces it.
- Pixel-art upscaling / depixelizing tracers (surveyed, rejected as primary).
