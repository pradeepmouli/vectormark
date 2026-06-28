# Seam-Graph (Shared-Edge Planar Map) — Design

## Problem

The AA-coverage work makes adjacent regions' seam arcs **point-identical at extraction**
(`φ_B = −φ_A` exactly, tested to 1e-9). But each region's contour is then **fit
independently**, so the two sides of a seam diverge → thin slivers (interior, and where a seam
reaches the silhouette, an outer-boundary notch). The coverage guarantee is necessary but does
not survive the fit stage.

## Goal

Make the rendered output gap-free **through the fit stage** by fitting each shared seam **once**
and having both adjacent regions reference the identical fitted edge. Because coverage already
makes the seam arcs coordinate-identical, the shared edge is *already computed* — this is
mostly bookkeeping (dedupe the two identical arcs into one fitted edge), not new geometry.

## Strategy: shared-edge (planar map)

Chosen over spread-and-stack (overlap/z-order trapping) because the coverage work already hands
us point-identical seams, making the exact approach cheap; shared-edge has **zero overdraw** and
exact boundaries. Spread-and-stack is documented as the **fallback** if junction handling proves
too gnarly in implementation.

## The five-step pipeline

1. **Extract** per-region contours from coverage (existing `significant_contours`). Already
   point-identical along seams.
2. **Classify** each contour point: *boundary* (region↔background), *seam* (region↔exactly one
   neighbor), or *junction-incident* (near where ≥3 regions meet). Side-region identity comes
   from the soft label field `L` (nearest other label across the contour).
3. **Build the edge graph:** nodes = junction points + boundary loop-closure points; edges =
   maximal arcs between nodes, each tagged with the ≤2 regions it borders. **Seam matching is
   exact-coordinate dedup** (the two regions' seam arcs are identical to ~1e-9), not fuzzy
   geometry — find the longest shared coordinate run (one reversed) between neighbor contours.
   Store each seam edge **once**.
4. **Fit each edge once** as an **open arc with fixed endpoints** (the junction nodes), reusing
   `fit_cubics` (follow-up #2) with endpoint tangents pinned so adjacent edges meet exactly at
   the shared node. Boundary edges fit the same way. A seam-free isolated region's single loop
   may still be recognized as a **primitive** (circle/ellipse/rect).
5. **Reassemble** each region as an ordered closed loop of its edges (forward/reversed),
   referencing the shared fitted geometry; holes → compound path (multiple `M...Z` subpaths,
   evenodd). Adjacent regions now share point-identical edges → gap-free, zero overdraw.

## The hard part: junction snapping

The gap-free identity was proven only for **2-color seams**. At a ≥3-region **junction** the
inverse-ΔE fallback means the three 0.5-crossings do **not** coincide exactly. So the graph
builder must:

- **Detect junctions:** pixels/contour points where ≥3 distinct region labels are mutually near
  (the soft field's top-3 are comparable — reuse the `soft_label_field` junction band).
- **Compute one junction node** per cluster (centroid of the incident crossing points), and
  **snap** all incident edge endpoints to it. Edges thus share an exact node even though their
  raw crossings differed by sub-pixel amounts.
- Edge tangents at a snapped junction are taken from each edge's own arc (not forced equal) —
  only the *position* is shared, so each region keeps its true edge direction into the node.

This is the one piece with no free lunch from the coverage work.

## No hybrids (primitive vs assembly)

A region is **either** a clean primitive (`<circle>`/`<rect>`/`<ellipse>`) **or** a fully
edge-assembled path — never a primitive with an edge grafted through it.

- **Seam-free isolated region** (boundary loop only) → primitive recognition applies as today
  → clean `<circle>` etc.
- **Any region sharing a seam** → edge-assembled path. Its free (boundary-vs-background) edges
  are fit as smooth arcs via `fit_cubics`, so a near-circular seam-bearing region still reads as
  round (a smooth arc with a genuine flat where the neighbor abuts), **not** a faceted polyline
  through a circle.

## Open-arc fitting (the fit-layer change)

`fit_cubics` already takes explicit endpoints + tangents (designed for this in follow-up #2).
Add a thin `fit_edge(arc_points, *, t0=None, t3=None)` wrapper that fits an **open** polyline
between fixed endpoints (estimating end tangents from the arc when not pinned), returning a path
fragment (no `M`/`Z`). Region assembly concatenates fragments into the closed `d`.

## Determinism

- Seam matching: exact-coordinate, deterministic.
- Junction clustering: deterministic (sorted by position, fixed nearness threshold).
- Edge/region ordering: value-ordered (sort by start coordinate). No RNG.

## Scope / sequencing

Depends on **follow-up #2 (cubics)** for `fit_cubics` open-arc fitting. Independent of
conditioning. Touches: new `seamgraph.py` (graph build + junction snap + reassembly),
`fit.py`/`_fitcurve.py` (`fit_edge`), `selector.py`/`pipeline.py` (route seam-bearing regions
through the graph; isolated regions keep the current primitive/path path), `contour.py`
(expose per-contour side-label classification).

## Validation

- **Gap-free render:** a synthetic two-region shared-seam image renders with **zero** interior
  background pixels along the seam (the sentinel-background test, 0 uncovered interior px).
- **Junction:** a three-region triple-point renders with no sliver at the junction (the
  Strategy-2 trigger now *passes*, not just survives).
- **Isolated primitive preserved:** a lone disk still emits `<circle>` (no regression from the
  graph path).
- **Seam-bearing near-circle:** renders as a smooth arc-path (curve commands, not a many-vertex
  polygon) — asserts "no polyline through a circle."
- **V-bird (conditioned):** interior + bottom-boundary gaps drop to ~0; structure preserved.
- **Determinism:** byte-identical across runs.
- **Fallback noted:** if junction snapping cannot reach zero-sliver in implementation, fall back
  to spread-and-stack (z-order overlap, trap width ≈ fit epsilon) for junction-incident regions
  only — documented, not built unless needed.
