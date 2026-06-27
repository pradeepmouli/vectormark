# Bounded Shape Grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make geometry fitting emit only *bounded simple shapes*, so a frayed, over-segmented boundary is unrepresentable and the clean shape wins — fixing serrated edges, hole-speckle, and shattered small features.

**Architecture:** Bound the two unbounded fitters (`fit_path` → ≤ segments, `recognize_polygon` → ≤ vertices), drop sub-threshold inner-contour holes, and wire `generate_geometry_candidates` to emit only bounded candidates plus a guaranteed-bounded "no-fit" fallback (logged via a strategy label). Selection stays the existing render-ΔE scorer; parsimony becomes the membership rule (the bound). Fills are untouched (decoupled in PR #40).

**Tech Stack:** Python 3.12+, numpy, scikit-image, pytest. Work in `.worktrees/shape-grammar` (branch `feat/bounded-shape-grammar`, on master `277d88f`).

## Global Constraints

- Python ≥ 3.12, pure-Python. DRY/YAGNI/TDD. Run tests with `uv run pytest`; use `rg` not `grep`.
- Bounds (named constants, starting values — CALIBRATED against the corpus in Task 5, not assumed): `MAX_PATH_SEGMENTS = 12`, `MAX_POLY_VERTICES = 10`, `HOLE_AREA_FRACTION = 0.01` (inner contour kept only if its area ≥ this × outer-contour area).
- A bounded fitter returns `None` (no candidate) when it cannot satisfy the bound — it must NEVER return an over-bound shape.
- The candidate set must never be empty for a real region: a guaranteed-bounded "no-fit" fallback (strategy label `NOFIT`) is emitted only when nothing else qualifies, so `select_geometry` always returns a shape.
- DO NOT touch fills (FlatFill/gradient/raster), the merge, or `fit_fill` — geometry only. Decomposition, performance, and raster are OUT OF SCOPE (separate follow-ups).
- Commit trailer EXACTLY, no other trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Do NOT `git add scratch/`.

## File Structure

- **Modify `src/vectormark/fit.py`** — add `MAX_PATH_SEGMENTS`/`MAX_POLY_VERTICES`; bound `fit_path` (count segments, `-> Shape | None`); default `recognize_polygon`'s `max_vertices` to the constant.
- **Modify `src/vectormark/contour.py`** — add `significant_contours(mask, *, min_hole_fraction)` that returns the outer contour + only inner contours clearing the area fraction.
- **Modify `src/vectormark/selection.py`** — add the `NOFIT` strategy label.
- **Modify `src/vectormark/selector.py`** — use `significant_contours`; guard the `PATH`/holed candidates on non-`None`; add the `NOFIT` loose-bounded fallback when the candidate set is otherwise empty.
- **Tests** — `tests/test_fit.py`, `tests/test_contour.py`, `tests/test_selector.py` (extend existing if present), plus a corpus/V-bird acceptance test.

---

### Task 1: Bound `fit_path` to a segment budget

**Files:**
- Modify: `src/vectormark/fit.py` (`fit_path`)
- Test: `tests/test_fit.py`

**Interfaces:**
- Produces: `MAX_PATH_SEGMENTS = 12`; `fit_path(contour, *, epsilon, max_error, max_segments=MAX_PATH_SEGMENTS) -> Shape | None` — returns the path `Shape` when it uses ≤ `max_segments` drawing commands (`L` + `Q`), else `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit.py` (create if absent, with `import numpy as np` and `from vectormark.fit import fit_path, MAX_PATH_SEGMENTS`):

```python
import numpy as np
from vectormark.fit import fit_path, MAX_PATH_SEGMENTS


def _square(n=40):
    # a clean axis-aligned square contour (closed ring), well under the segment budget
    top = [(x, 0) for x in range(n)]
    right = [(n - 1, y) for y in range(n)]
    bot = [(x, n - 1) for x in range(n - 1, -1, -1)]
    left = [(0, y) for y in range(n - 1, -1, -1)]
    return np.array(top + right + bot + left + [top[0]], float)


def _noisy_blob(n=200, seed=0):
    # a jagged closed contour that needs many segments -> must exceed the budget
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    r = 50 + rng.normal(0, 6, n)              # heavy per-vertex radial noise
    pts = np.column_stack([60 + r * np.cos(t), 60 + r * np.sin(t)])
    return np.vstack([pts, pts[0]])


def test_clean_shape_fits_within_budget():
    shape = fit_path(_square(), epsilon=1.5, max_error=1.0)
    assert shape is not None
    d = shape.params["d"]
    assert d.count("L") + d.count("Q") <= MAX_PATH_SEGMENTS


def test_frayed_contour_exceeds_budget_returns_none():
    # with a tight max_error the jagged blob needs > MAX_PATH_SEGMENTS quadratics
    assert fit_path(_noisy_blob(), epsilon=0.5, max_error=0.5) is None


def test_explicit_low_budget_rejects():
    assert fit_path(_square(), epsilon=1.5, max_error=1.0, max_segments=2) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fit.py -k "budget or frayed or fits_within" -v`
Expected: FAIL — `fit_path` has no `max_segments` and never returns `None` (and `MAX_PATH_SEGMENTS` undefined).

- [ ] **Step 3: Implement the bound**

In `src/vectormark/fit.py`, add the constant near the top (after imports):

```python
MAX_PATH_SEGMENTS = 12   # a "shape" is simple; a path needing more drawing commands than
                         # this is fraying (tracing AA noise), not a shape -> disqualified.
```

Rewrite `fit_path` to count drawing commands and bail when over budget (keep the existing corner-split logic; only add the counter + the `max_segments` param + the `-> Shape | None` return):

```python
def fit_path(contour: np.ndarray, *, epsilon: float, max_error: float,
             max_segments: int = MAX_PATH_SEGMENTS) -> Shape | None:
    """Corner-split the contour; emit lines for straight runs, quadratic Béziers otherwise.
    Returns None if the result needs more than `max_segments` drawing commands — a frayed
    boundary is not a simple shape and must not be emitted."""
    pts = np.asarray(contour, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    ring = pts[:-1] if closed else pts
    simp = rdp(ring, epsilon)
    corners = corner_indices(np.vstack([simp, simp[0]]), angle_threshold_deg=40)
    corner_pts = simp[corners] if corners else simp[[0]]
    cut_idx = sorted({int(np.argmin(np.hypot(*(ring - cp).T))) for cp in corner_pts})
    if len(cut_idx) < 2:
        cut_idx = [0, len(ring) // 2]

    d = f"M{_fmt(ring[cut_idx[0]][0])} {_fmt(ring[cut_idx[0]][1])} "
    segs = 0
    for k in range(len(cut_idx)):
        i0 = cut_idx[k]
        i1 = cut_idx[(k + 1) % len(cut_idx)]
        seg = ring[i0:i1 + 1] if i1 > i0 else np.vstack([ring[i0:], ring[: i1 + 1]])
        if len(seg) < 2:
            continue
        if _segment_is_straight(seg, epsilon):
            d += f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
            segs += 1
        else:
            for b in fit_quadratic_beziers(seg, max_error):
                d += f"Q{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
                segs += 1
        if segs > max_segments:
            return None
    d += "Z"
    return Shape("path", {"d": d})
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_fit.py -k "budget or frayed or fits_within" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/fit.py tests/test_fit.py
git commit -m "feat(fit): bound fit_path to MAX_PATH_SEGMENTS (frayed boundary -> None)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Wire `MAX_POLY_VERTICES` into `recognize_polygon`

**Files:**
- Modify: `src/vectormark/fit.py` (`recognize_polygon`)
- Test: `tests/test_fit.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `MAX_POLY_VERTICES = 10`; `recognize_polygon(contour, *, epsilon, max_vertices=MAX_POLY_VERTICES) -> Shape | None` — already returns `None` when the simplified polygon exceeds `max_vertices`; this names the default as the grammar bound.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit.py`:

```python
from vectormark.fit import recognize_polygon, MAX_POLY_VERTICES


def test_polygon_default_bound_is_the_constant():
    # a clean hexagon (6 verts) is within the bound -> recognized
    import numpy as np
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexa = np.column_stack([50 + 40 * np.cos(t), 50 + 40 * np.sin(t)])
    ring = np.vstack([hexa, hexa[0]])
    shp = recognize_polygon(ring, epsilon=1.5)
    assert shp is not None and 3 <= len(shp.params["points"]) <= MAX_POLY_VERTICES
    # forcing a 2-vertex bound rejects it
    assert recognize_polygon(ring, epsilon=1.5, max_vertices=2) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fit.py -k polygon_default_bound -v`
Expected: FAIL — `MAX_POLY_VERTICES` is undefined.

- [ ] **Step 3: Implement**

In `src/vectormark/fit.py`, add the constant near `MAX_PATH_SEGMENTS`:

```python
MAX_POLY_VERTICES = 10   # a polygon "shape" has few vertices; more is a traced jagged edge.
```

Change `recognize_polygon`'s signature default from `max_vertices: int = 8` to `max_vertices: int = MAX_POLY_VERTICES` (body unchanged — it already returns `None` when `not (3 <= len(simp) <= max_vertices)`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fit.py -k polygon_default_bound -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/fit.py tests/test_fit.py
git commit -m "feat(fit): name MAX_POLY_VERTICES as the polygon grammar bound

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Drop sub-threshold inner-contour holes

**Files:**
- Modify: `src/vectormark/contour.py`
- Test: `tests/test_contour.py`

**Interfaces:**
- Consumes: existing `region_contours(mask)` (returns outer + all hole contours), `_polygon_area`.
- Produces: `HOLE_AREA_FRACTION = 0.01`; `significant_contours(mask, *, min_hole_fraction=HOLE_AREA_FRACTION) -> list[np.ndarray]` — the largest-area (outer) contour plus only those inner contours whose area ≥ `min_hole_fraction × outer_area`. A region whose only holes are sub-threshold noise returns a single (outer) contour → it fills solid, no even-odd speckle.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_contour.py` (create if absent):

```python
import numpy as np
from vectormark.contour import significant_contours, region_contours


def _disc_with_hole(R=40, r_hole=2, big_hole=False):
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    mask = ((yy - 50) ** 2 + (xx - 50) ** 2) <= R ** 2
    rh = 15 if big_hole else r_hole
    mask &= ~(((yy - 50) ** 2 + (xx - 50) ** 2) <= rh ** 2)
    return mask


def test_tiny_noise_hole_is_dropped():
    mask = _disc_with_hole(r_hole=2)               # a 2px speck hole
    assert len(region_contours(mask)) >= 2          # raw: outer + speck
    assert len(significant_contours(mask)) == 1      # filtered: outer only


def test_genuine_counter_is_kept():
    mask = _disc_with_hole(big_hole=True)           # a real 15px counter
    out = significant_contours(mask)
    assert len(out) == 2                            # outer + the genuine hole
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_contour.py -k "noise_hole or counter" -v`
Expected: FAIL — `significant_contours` does not exist.

- [ ] **Step 3: Implement**

In `src/vectormark/contour.py`, add (reuse the existing `region_contours` and `_polygon_area`):

```python
HOLE_AREA_FRACTION = 0.01   # an inner contour smaller than this fraction of the outer
                            # area is quantization speckle, not an intentional counter.


def significant_contours(mask: np.ndarray, *, min_hole_fraction: float = HOLE_AREA_FRACTION):
    """Outer contour + only inner contours that clear the area fraction. Drops tiny
    noise holes (the white specks) while keeping genuine counters."""
    contours = [c for c in region_contours(mask) if len(c) >= 3]
    if not contours:
        return []
    areas = [_polygon_area(c) for c in contours]
    outer_i = int(np.argmax(areas))
    outer_area = areas[outer_i] or 1.0
    keep = [contours[outer_i]]
    floor = min_hole_fraction * outer_area
    keep += [c for i, c in enumerate(contours) if i != outer_i and areas[i] >= floor]
    return keep
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_contour.py -k "noise_hole or counter" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/contour.py tests/test_contour.py
git commit -m "feat(contour): significant_contours drops sub-threshold noise holes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Wire bounded candidates + the no-fit fallback

**Files:**
- Modify: `src/vectormark/selection.py` (add `NOFIT` label)
- Modify: `src/vectormark/selector.py` (`generate_geometry_candidates`)
- Test: `tests/test_selector.py`

**Interfaces:**
- Consumes: `fit_path` (now `Shape | None`, Task 1), `recognize_polygon` (Task 2), `significant_contours` (Task 3), `contour.rdp`.
- Produces: `NOFIT = "nofit"`; `generate_geometry_candidates(...)` emits only bounded candidates, guards `PATH`/holed on non-`None`, and — only when no other candidate qualifies — appends one guaranteed-bounded `NOFIT` polygon (rdp forced to ≤ `MAX_POLY_VERTICES` vertices) so the set is never empty and the no-fit region is visible in the report's strategy histogram.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_selector.py` (create if absent):

```python
import numpy as np
from vectormark.selector import generate_geometry_candidates
from vectormark.selection import PATH, NOFIT, PRIMITIVE, POLYGON
from vectormark.types import Region


class _Opt:
    epsilon = 1.5
    max_error = 1.0


def _region_from_mask(mask):
    return Region(label=1, mask=mask, color_hex="#000000")


def _square_mask(n=60, pad=20):
    m = np.zeros((n + 2 * pad, n + 2 * pad), bool)
    m[pad:pad + n, pad:pad + n] = True
    return m


def _frayed_mask(seed=0):
    H = W = 140
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:H, :W]
    base = ((yy - 70) ** 2 + (xx - 70) ** 2) <= 45 ** 2
    noise = rng.random((H, W)) < 0.5
    ring = (((yy - 70) ** 2 + (xx - 70) ** 2) <= 50 ** 2) & ~base
    return base | (ring & noise)                    # a heavily-jagged blob edge


def test_clean_region_has_no_nofit():
    cands = generate_geometry_candidates(_region_from_mask(_square_mask()), _Opt(), None, 0.0)
    strategies = {c.strategy for c in cands}
    assert NOFIT not in strategies                  # a square fits a real bounded shape
    assert strategies & {PRIMITIVE, POLYGON, PATH}  # at least one real grammar member


def test_frayed_region_falls_back_to_bounded_nofit():
    cands = generate_geometry_candidates(_region_from_mask(_frayed_mask()), _Opt(), None, 0.0)
    assert cands, "candidate set must never be empty"
    # the frayed blob cannot be a real bounded shape -> NOFIT fallback, still bounded
    nofit = [c for c in cands if c.strategy == NOFIT]
    if nofit:
        pts = nofit[0].shape.params["points"]
        from vectormark.fit import MAX_POLY_VERTICES
        assert 3 <= len(pts) <= MAX_POLY_VERTICES
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_selector.py -k "nofit or no_nofit or bounded" -v`
Expected: FAIL — `NOFIT` is undefined and the no-fit fallback isn't wired.

- [ ] **Step 3: Add the `NOFIT` label**

In `src/vectormark/selection.py`, beside the other labels (e.g. after `PATH = "path"`), add:

```python
NOFIT = "nofit"            # loosest bounded fallback when no grammar member fits (logged)
```

- [ ] **Step 4: Implement the wiring**

In `src/vectormark/selector.py`:

Add a helper (above `generate_geometry_candidates`):

```python
def _loose_bounded_polygon(contour: np.ndarray, max_vertices: int, epsilon: float) -> Shape:
    """A guaranteed ≤ max_vertices polygon by increasing rdp tolerance until it fits.
    The no-fit floor: clean but approximate, never frayed."""
    from .contour import rdp
    eps = max(epsilon, 0.5)
    pts = np.asarray(contour, float)
    for _ in range(20):
        simp = rdp(pts, eps)
        if np.allclose(simp[0], simp[-1]):
            simp = simp[:-1]
        if 3 <= len(simp) <= max_vertices:
            return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})
        eps *= 1.6
    simp = simp[:max_vertices] if len(simp) > max_vertices else simp
    return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})
```

Import `MAX_POLY_VERTICES` and replace the unconditional `PATH` append (the tail block) so the path candidate is guarded and a `NOFIT` fallback fires when the set is empty:

```python
    if axis is None or not has_symmetry_preserving:
        gpoly = recognize_polygon(contour, epsilon=opt.epsilon)
        if gpoly is not None:
            cands.append(GeomCandidate(POLYGON, gpoly))
        gpath = fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error)
        if gpath is not None:
            cands.append(GeomCandidate(PATH, gpath))

    if not cands:
        cands.append(GeomCandidate(
            NOFIT, _loose_bounded_polygon(contour, MAX_POLY_VERTICES, opt.epsilon)))

    return cands
```

Update the imports at the top of `selector.py`: add `NOFIT` to the `from .selection import (...)` list, and add `MAX_POLY_VERTICES` to `from .fit import (...)`.

> Note for the implementer: the holed-path branch (`if len(contours) > 1:`) must use `significant_contours(region.mask)` instead of `region_contours(region.mask)` at the contour-gathering line (`contours = [c for c in region_contours(region.mask) if len(c) >= 3]`) so dropped noise holes never produce an even-odd candidate. Replace that one call with `significant_contours` (it already filters `len(c) >= 3` and the hole area). Also guard any `fit_path(...)` call inside the holed branch on non-`None` the same way (skip the holed candidate if its path is over budget — the `NOFIT` fallback covers it).

- [ ] **Step 5: Run the focused tests + full suite**

Run: `uv run pytest tests/test_selector.py -k "nofit or no_nofit or bounded" -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: some existing geometry/golden tests may shift because the frayed `PATH` is now disqualified and noise holes are dropped — this is the intended behavior change. Update goldens/assertions that encoded the OLD frayed output to the new bounded output; do NOT weaken a test to pass — re-derive its expected value from the new (cleaner) geometry, and note each changed golden in the commit message. Genuinely-unrelated failures are bugs to fix.

- [ ] **Step 6: Commit**

```bash
git add src/vectormark/selector.py src/vectormark/selection.py tests/
git commit -m "feat(selector): emit only bounded candidates + NOFIT fallback; drop noise holes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Corpus calibration + V-bird acceptance

**Files:**
- Modify: calibration constants in `src/vectormark/fit.py` / `src/vectormark/contour.py` only if the corpus shows it
- Test: `tests/test_bounded_grammar_acceptance.py`

**Interfaces:**
- Consumes: the full bounded pipeline (Tasks 1–4).
- Context: corpus at `/Users/pmouli/GitHub.nosync/active/py/vectormark/scratch/real-logos/` (UNTRACKED — never `git add scratch/`). V-bird at `/Users/pmouli/.claude/image-cache/765f9098-2b6a-45f9-b368-3394ea44b5c0/3.png` if present.

- [ ] **Step 1: Write the acceptance test (V-bird + bound invariant)**

Create `tests/test_bounded_grammar_acceptance.py`:

```python
import os
import re
import numpy as np
import pytest
from PIL import Image
from vectormark.pipeline import idealize, Options, _flatten_on_white
from vectormark.fit import MAX_PATH_SEGMENTS

VBIRD = os.path.expanduser("~/.claude/image-cache/765f9098-2b6a-45f9-b368-3394ea44b5c0/3.png")


def _max_path_segments(svg: str) -> int:
    worst = 0
    for d in re.findall(r'd="([^"]*)"', svg):
        worst = max(worst, d.count("L") + d.count("Q"))
    return worst


@pytest.mark.skipif(not os.path.exists(VBIRD), reason="V-bird image not present")
def test_vbird_paths_are_bounded_and_unspeckled():
    arr = _flatten_on_white(Image.open(VBIRD))
    svg = idealize(arr, options=Options(max_colors=16))
    # no emitted path exceeds the segment budget (no fraying)
    assert _max_path_segments(svg) <= MAX_PATH_SEGMENTS
    # no even-odd speckle holes
    assert "evenodd" not in svg or svg.count("evenodd") <= 1
    # the dots came through as circles (fit-to-evidence), not eroded paths
    assert svg.count("<circle") >= 1 or svg.count("<ellipse") >= 1


def test_clean_flat_square_unchanged():
    img = np.full((80, 80, 3), 255, np.uint8)
    img[20:60, 20:60] = (200, 40, 40)
    svg = idealize(img, options=Options(max_colors=16))
    assert _max_path_segments(svg) <= MAX_PATH_SEGMENTS
    assert "<rect" in svg or "<polygon" in svg or "<path" in svg
```

- [ ] **Step 2: Run to verify the bound invariant holds (or reveals calibration need)**

Run: `uv run pytest tests/test_bounded_grammar_acceptance.py -v`
Expected: the square test PASSES; the V-bird test PASSES if the image is present. If the V-bird test FAILS on the `<circle>` assertion, the dots aren't being recognized as primitives — investigate primitive recognition before tuning bounds.

- [ ] **Step 3: Corpus regression — confirm good logos aren't degraded**

Run this comparison harness (NOT committed; reads the untracked corpus) and record before/after per logo:

```bash
uv run python - <<'PY'
import glob, os, re
from PIL import Image
from vectormark.pipeline import idealize, Options, _flatten_on_white
C="/Users/pmouli/GitHub.nosync/active/py/vectormark/scratch/real-logos"
def maxseg(svg):
    return max([d.count("L")+d.count("Q") for d in re.findall(r'd="([^"]*)"', svg)] or [0])
for f in sorted(glob.glob(C+"/*.png")):
    if os.path.basename(f).startswith(("cmp_","out_","poc_","contact_","_tri")): continue
    svg, rep = idealize(_flatten_on_white(Image.open(f)), options=Options(max_colors=16), report=True)
    nofit = rep.strategies.get("nofit", 0)          # NOFIT regions = decomposition candidates
    print(f"{os.path.basename(f):20} maxseg={maxseg(svg):3d} paths={svg.count('<path')} "
          f"circ={svg.count('<circle')} evenodd={svg.count('evenodd')} nofit={nofit}")
PY
```

Acceptance: every logo's `maxseg ≤ MAX_PATH_SEGMENTS`; no logo that previously rendered cleanly now shows a `NOFIT` strategy in its `idealize(report=True)` strategy histogram (re-run with `report=True` for any suspicious logo). If a genuinely-curvy-but-simple logo (e.g. a script wordmark) newly hits `NOFIT`, **raise `MAX_PATH_SEGMENTS`** (e.g. 12 → 16) and re-run, rather than letting it degrade — record the final value and the logo that drove it.

- [ ] **Step 4: Calibrate the constants if the corpus shows it**

Adjust `MAX_PATH_SEGMENTS` / `MAX_POLY_VERTICES` (`fit.py`) and `HOLE_AREA_FRACTION` (`contour.py`) only as Step 3 dictates. After each change re-run `uv run pytest tests/test_fit.py tests/test_contour.py tests/test_selector.py tests/test_bounded_grammar_acceptance.py -q` and the Step-3 harness. Document the final values + the corpus evidence in the commit message. List any logo flagged as a `NOFIT` (decomposition candidate) for the fast-follow.

- [ ] **Step 5: Run full suite + commit**

Run: `uv run pytest -q`
Expected: green.

```bash
git add tests/test_bounded_grammar_acceptance.py src/vectormark/fit.py src/vectormark/contour.py
git commit -m "test(grammar): V-bird + corpus acceptance; calibrate bounds

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: `uv run pytest -q` green; dispatch a whole-branch review (most-capable model) over `master..HEAD` — pay special attention to the changed goldens (verify each new expected value is the *cleaner* geometry, not a weakened assertion). Then superpowers:finishing-a-development-branch to open the PR; record any `NOFIT` corpus marks as the decomposition fast-follow. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
