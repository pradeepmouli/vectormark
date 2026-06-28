# Seam-Graph (Shared-Edge Planar Map) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Make adjacent regions gap-free *through the fit stage* by fitting each shared seam once and having both regions reference the identical fitted edge.

**Architecture:** Two phases. **Phase A** builds `seamgraph.py` as a pure, synthetically-validated module: classify contour points by the region across them, build a planar edge graph (seam dedup by coordinate identity — validated: adjacent coverage contours share 40+ point-identical samples per seam), snap junction clusters to one node, fit each edge once as an open arc (reusing `fit_cubic_beziers`), reassemble each region as an ordered loop of shared edges (compound if holed). **Phase B** (separate plan, after A proves out) routes flat seam-bearing regions through the module in the pipeline; occlusion, symmetry pairs, and isolated-region primitives keep their existing paths.

**Tech Stack:** numpy, scipy.ndimage; reuses `softlabel.soft_label_field`/`region_coverage`, `_fitcurve.fit_cubic_beziers`.

## Global Constraints

- Python ≥ 3.12, pure-Python. TDD. `rg` not `grep`. Determinism: exact-coordinate matching, junction clustering sorted by position, edge/region ordering value-ordered — no RNG.
- **No hybrids:** a region is either a clean primitive (isolated) or a fully edge-assembled path (seam-bearing) — never a primitive with an edge grafted through it. (Enforced in Phase B; Phase A only assembles paths.)
- **Gap-free is the invariant:** both regions of a seam must reference the byte-identical fitted edge geometry.
- Do NOT `git add scratch/`. Commit trailer EXACTLY: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Pre-existing missing-optional-dep collection errors (`mcp`, `httpx`) and the lone resvg-fallback-test warning are environmental, not yours.

---

## PHASE A — standalone `seamgraph.py` module

### Task 1: Contour side-label classification

**Files:**
- Create: `src/vectormark/seamgraph.py`
- Test: `tests/test_seamgraph.py`

**Interfaces:**
- Produces: `classify_contour(contour, L, region_idx, *, bg_idx) -> np.ndarray` — for each point of `contour` (shape (N,2), (x,y)), the integer label of the region/background **across** the boundary (the argmax of `L` at that sub-pixel point among labels ≠ `region_idx`), or `bg_idx` for background. Returns (N,) int array.

- [ ] **Step 1: failing test** — synthetic 3-region image (A/B/C meeting at a junction, as in the validated prototype). Build `L`; for region A's contour, assert the points along the A|B seam classify as B's index, along A|C as C's, and the outer edge as `bg_idx`. (Use a tolerance: ≥80% of seam-band points carry the expected neighbor label.)
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement.** Bilinearly sample `L` at each contour point (points sit at pixel-edge midpoints — sample the two adjacent pixel centers and take the max-competing label among `j ≠ region_idx`). Map background palette row → `bg_idx`. Value-ordered argmax ties.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: commit** `feat(seamgraph): classify_contour — label the region across each boundary point`.

### Task 2: Planar edge graph (seam dedup by coordinate identity)

**Files:** Modify `src/vectormark/seamgraph.py`; extend `tests/test_seamgraph.py`.

**Interfaces:**
- Consumes: `classify_contour` (Task 1).
- Produces:
  - `EdgeGraph` dataclass: `nodes: list[tuple[float,float]]`, `edges: list[Edge]` where `Edge` has `pts: np.ndarray`, `region_a: int`, `region_b: int | None` (None = background side), `node0: int`, `node1: int`.
  - `build_graph(region_contours: dict[int, np.ndarray], L, *, bg_idx) -> EdgeGraph` — classify each region's contour, split each contour into maximal runs of constant (this_region, other_label), **dedup** runs shared by two regions by exact-coordinate match (the A-side run and the B-side run are the same points reversed → store ONE `Edge` with both region ids), set boundary runs' `region_b=None`. Endpoints become graph nodes.

- [ ] **Step 1: failing test** — on the 3-region synthetic: assert `build_graph` yields exactly the 3 seam edges (each with both region ids set) + the outer boundary edges (region_b None); assert each seam edge's `pts` equals the matching run in BOTH source contours (coordinate-identical). Assert no seam edge is duplicated.
- [ ] **Step 2:** FAIL. → **Step 3:** implement (runs via `classify_contour` label transitions; dedup via a coordinate-keyed dict rounding to 1e-6). → **Step 4:** PASS. → **Step 5: commit** `feat(seamgraph): build planar edge graph with exact-coordinate seam dedup`.

### Task 3: Junction snapping

**Files:** Modify `seamgraph.py`; extend tests.

**Interfaces:**
- Produces: `snap_junctions(graph, *, reach=1.5) -> EdgeGraph` — cluster node positions within `reach` px (where ≥3 edges incident, i.e. a triple point), replace each cluster with one node at its centroid, and move every incident edge endpoint to that centroid (adjust the edge's first/last `pts` row). Edge tangents are NOT forced equal — only the shared endpoint position.

- [ ] **Step 1: failing test** — the prototype showed the 3 seams converge near (59.75,59.75) but end at (59.5,59)/(59.5,60)/(60,59.5). Assert after `snap_junctions` all three seam edges share ONE identical junction node coordinate, and that node ≈ (59.75,59.75) within 1px. Assert non-junction endpoints are unchanged.
- [ ] **Step 2:** FAIL → **Step 3:** implement (group endpoints by spatial proximity via `scipy.ndimage`/a deterministic union-find over sorted nodes; centroid; rewrite incident edge endpoints). → **Step 4:** PASS. → **Step 5: commit** `feat(seamgraph): snap junction clusters to a shared node`.

### Task 4: Open-arc edge fitting (`fit_edge`)

**Files:** Modify `src/vectormark/fit.py` (add `fit_edge`); test in `tests/test_fit.py`.

**Interfaces:**
- Consumes: `fit_cubic_beziers` (PR #44).
- Produces: `fit_edge(pts, *, epsilon, max_error) -> str` — fit an OPEN polyline between its fixed endpoints into an SVG path fragment (no `M`/`Z`): straight runs → `L`, curved runs → `C` (denoise + cubic, as `fit_path` does), endpoints preserved exactly. Returns the concatenated command string starting after the first point (caller emits the `M`).

- [ ] **Step 1: failing test** — `fit_edge` on a straight polyline → only `L`, ending exactly at the last point; on an arc polyline → contains `C`, first/last coordinates equal the input endpoints (so adjacent edges meet). → **Steps 2-4** as usual. → **Step 5: commit** `feat(fit): fit_edge — open-arc fitting with fixed endpoints for shared seams`.

### Task 5: Region reassembly → path `d`

**Files:** Modify `seamgraph.py`; extend tests.

**Interfaces:**
- Produces: `region_path_d(graph, region_idx, *, epsilon, max_error) -> str` — collect the edges bordering `region_idx`, order them head-to-tail into closed loop(s) (multiple loops → holes → compound `d` with several `M...Z`, evenodd), fit each edge ONCE via `fit_edge` (cache by edge identity so a seam shared with a neighbor fits identically), and emit the `d` string. Reversed edges reuse the same fitted fragment reversed.

- [ ] **Step 1: failing test** — on the 3-region synthetic, build graph → snap → `region_path_d` for A, B, C. Assert: (a) each is a valid closed `d` (`M`…`Z`); (b) **the seam fragment is byte-identical** between the two regions that share it (the gap-free invariant — fit once, both reference it); (c) rasterizing all three over a magenta background leaves **zero** interior background pixels along the seams (the gap-free render test, via `tests/_render`). → **Steps 2-4.** → **Step 5: commit** `feat(seamgraph): reassemble regions from shared fitted edges (gap-free)`.

### Task 6 (Phase A acceptance): module-level gap-free + junction + determinism

**Files:** `tests/test_seamgraph_acceptance.py`.

- [ ] Gap-free: two-region and three-region synthetics render with 0 interior background px along seams.
- [ ] Junction: the triple-point renders with no sliver at the junction node (0 uncovered px in a 3px box around it).
- [ ] Determinism: `region_path_d` byte-identical across runs.
- [ ] Shared-fragment identity: every seam's fitted fragment is identical between its two regions (assert across all seams).
- [ ] Commit `test(seamgraph): phase-A acceptance — gap-free seams, snapped junctions, determinism`.

---

## PHASE B — pipeline integration (SEPARATE plan, authored after Phase A is green)

Outline (not yet task-decomposed; detailed once the module proves out):
- Retain/recompute the soft field `L` at geometry-fit time (extend `attach_coverage_field` to also return `(L, idx_map, bg_idx)` or stash on a context).
- In `build_candidates`/`_render_body`: partition the flat regions of a component into **isolated** (no seam — keep current primitive/path selection) vs **seam-bearing** (route through `seamgraph`). Build ONE graph per component from the seam-bearing regions' contours; emit each via `region_path_d`.
- Leave occlusion (`reconstruct_scene`, ScenePrimitive z-order) and symmetry (straddlers/pairs) on their existing paths in v1; document that seam-graph applies to the non-occluded, non-symmetric flat partition first.
- Acceptance: V-bird (conditioned) interior + bottom-boundary gaps → ~0; isolated dots still `<circle>`; seam-bearing near-circle renders as a smooth arc-path (no polyline-through-circle); corpus non-regression; determinism.
- Fallback (documented, build only if junction snapping can't reach zero-sliver in real logos): spread-and-stack (z-order overlap, trap width ≈ fit epsilon) for junction-incident regions only.
