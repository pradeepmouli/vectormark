# Antialiasing-Aware Sub-Pixel Boundary Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract each region's boundary from the soft pre-quantization antialiased signal (a per-region coverage field) instead of the binary mask, so contours are smooth by construction with adjacent regions sharing an identical sub-pixel seam (no gaps/overlaps).

**Architecture:** Build a global partition-of-unity soft label field once in `_segment_image`; derive each region's signed-margin coverage (`φ = L_k − max_{j≠k}L_j`, mapped to `cov = (φ+1)/2`, boundary at 0.5) from that ONE shared field — which makes shared seams identical by float-negation identity. Add an optional `coverage` field to `Region`; `contour.region_contours`/`outer_contour` switch their `find_contours` source to it (level stays 0.5); `coverage=None` is byte-identical to today. The bool `mask` is unchanged — symmetry/occlusion/score/surface_merge keep using it.

**Tech Stack:** Python 3.12+, numpy, scikit-image (`find_contours`, `label`), pytest. Work in `.worktrees/aa-contours` (branch `feat/aa-contours`, on master `49b29e9`).

## Global Constraints

- Python ≥ 3.12, pure-Python, deterministic (no RNG; ties value-ordered via `np.lexsort`, mirroring `color.extract_palette`). Run tests with `uv run pytest`; use `rg` not `grep`.
- **The no-gap shared-seam guarantee is the central, non-negotiable requirement.** Every region's coverage is derived from ONE global soft label field `L` (never re-derived per-region from only its own pixels), so along an A|B seam `φ_B = −φ_A` exactly.
- The bool `mask` STAYS on `Region` and all existing consumers (symmetry.py, occlusion.py, score.py, surface_merge.py) keep using it untouched. ONLY contour extraction reads `coverage`.
- `coverage=None` ⇒ `region_contours`/`outer_contour` produce byte-for-byte today's bilevel output (backward compat).
- The coverage boundary sits at **0.5**, so `contour.py`'s existing `find_contours(..., 0.5)` level is UNCHANGED.
- Background is a palette LABEL in the soft field (the mark-vs-background edge is the most common seam) even though `segment` excludes it as a Region.
- OUT OF SCOPE: changing `quantize`/the palette; changing the fitters/scorer or the bounded-grammar RMS threshold; the bool-mask consumers; gradient seam logic (`surface_merge`). The full planar seam-graph (Strategy 2) is built ONLY if the junction test (Task 6) fails.
- Commit trailer EXACTLY, no other trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Do NOT `git add scratch/`.

## File Structure

- **Modify `src/vectormark/types.py`** — `Region` gains `coverage: np.ndarray | None = None`.
- **Modify `src/vectormark/contour.py`** — `region_contours`/`outer_contour`/`significant_contours`/`region_corner_radius` gain an optional `coverage` source.
- **Create `src/vectormark/softlabel.py`** — `alpha_unmix`, `soft_label_field`, `region_coverage`: the soft-field algorithm. Pure (no pipeline imports).
- **Modify `src/vectormark/pipeline.py`** (`_segment_image`) — compute the field once, attach `coverage` to each region; pass `region.coverage` at the `region_corner_radius` call site.
- **Modify `src/vectormark/selector.py`, `src/vectormark/occlusion.py`** — pass `region.coverage` to the contour calls.
- **Tests** — `tests/test_contour_coverage.py`, `tests/test_softlabel.py`, `tests/test_aa_contour_acceptance.py`.

---

### Task 1: `Region.coverage` field + contour source switch (backward-compatible)

**Files:**
- Modify: `src/vectormark/types.py` (`Region`)
- Modify: `src/vectormark/contour.py` (`region_contours`, `outer_contour`)
- Test: `tests/test_contour_coverage.py`

**Interfaces:**
- Produces: `Region.coverage: np.ndarray | None = None`; `region_contours(mask, *, coverage=None) -> list[np.ndarray]` and `outer_contour(mask, *, coverage=None) -> np.ndarray` — when `coverage` is given, `find_contours` runs on it (level 0.5) instead of `mask.astype(float)`; `coverage=None` is unchanged behavior.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_contour_coverage.py`:

```python
import numpy as np
from vectormark.contour import region_contours, outer_contour
from vectormark.types import Region


def _disc_mask(r=20, H=80):
    yy, xx = np.ogrid[:H, :H]
    return ((yy - 40) ** 2 + (xx - 40) ** 2) <= r ** 2


def test_coverage_none_is_byte_identical_to_mask():
    m = _disc_mask()
    a = region_contours(m)
    b = region_contours(m, coverage=None)
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca, cb)


def test_coverage_field_is_traced_smoother_than_binary():
    # a smooth coverage field (soft disc) yields a contour closer to the true circle
    H = 80
    yy, xx = np.ogrid[:H, :H]
    d = np.sqrt((yy - 40.0) ** 2 + (xx - 40.0) ** 2)
    cov = np.clip(0.5 + (20.0 - d), 0, 1)          # 0.5 isocontour at radius 20, smooth
    mask = d <= 20
    cs_cov = outer_contour(mask, coverage=cov)
    cs_bin = outer_contour(mask)
    def rms_radius_err(c):
        rr = np.hypot(c[:, 0] - 40.0, c[:, 1] - 40.0)
        return float(np.sqrt(np.mean((rr - 20.0) ** 2)))
    assert rms_radius_err(cs_cov) < rms_radius_err(cs_bin)   # smoother by construction
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_contour_coverage.py -v`
Expected: FAIL — `region_contours`/`outer_contour` take no `coverage` kwarg; `Region` has no `coverage`.

- [ ] **Step 3: Implement**

In `src/vectormark/types.py`, add the field to `Region` (keep `mask` first-class):

```python
@dataclass
class Region:
    """One connected, single-colour area of the quantised image."""
    label: int
    mask: np.ndarray          # bool (H, W) — all non-contour consumers use this
    color_hex: str
    coverage: np.ndarray | None = None   # float (H,W), region's α field, boundary at 0.5
```

In `src/vectormark/contour.py`, switch the source in `region_contours` and `outer_contour`:

```python
def region_contours(mask: np.ndarray, *, coverage: np.ndarray | None = None) -> list[np.ndarray]:
    """All sub-pixel contours (outer + holes), (N,2) (x,y), area-sorted (outer first).
    With `coverage` (a float α field, boundary at 0.5) the contour is extracted from the
    smooth field instead of the bilevel mask; `coverage=None` is the original behavior."""
    field = mask.astype(float) if coverage is None else coverage
    padded = np.pad(field, 1)
    contours = find_contours(padded, 0.5)
    out = [np.column_stack([c[:, 1] - 1, c[:, 0] - 1]) for c in contours]
    out.sort(key=_polygon_area, reverse=True)
    return out
```

Apply the identical `field = mask.astype(float) if coverage is None else coverage` switch to `outer_contour` (find its `np.pad(mask.astype(float), 1)` line and replace `mask.astype(float)` with `field`, adding the `*, coverage=None` param).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_contour_coverage.py -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS — no existing caller passes `coverage`, so behavior is unchanged. (The pre-existing `test_stdio_server_exposes_idealize_logo_tool` MCP-stdio failure, if it appears, is unrelated.)

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/types.py src/vectormark/contour.py tests/test_contour_coverage.py
git commit -m "feat(contour): optional coverage source for sub-pixel contour extraction

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `alpha_unmix` — coverage from a two-color blend

**Files:**
- Create: `src/vectormark/softlabel.py`
- Test: `tests/test_softlabel.py`

**Interfaces:**
- Produces: `alpha_unmix(rgb, c_a, c_b) -> np.ndarray` — for pixels `V` (`(...,3)` float) and two colors `c_a`,`c_b` (`(3,)` float), returns `α = clip((V−c_b)·(c_a−c_b)/|c_a−c_b|², 0, 1)` (α=1 ⇒ pure `c_a`). Vectorized over any leading shape.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_softlabel.py`:

```python
import numpy as np
from vectormark.softlabel import alpha_unmix


def test_pure_colors_give_0_and_1():
    ca = np.array([20.0, 40.0, 200.0]); cb = np.array([255.0, 255.0, 255.0])
    assert abs(alpha_unmix(ca, ca, cb) - 1.0) < 1e-9
    assert abs(alpha_unmix(cb, ca, cb) - 0.0) < 1e-9


def test_midpoint_gives_half():
    ca = np.array([0.0, 0.0, 0.0]); cb = np.array([255.0, 255.0, 255.0])
    mid = (ca + cb) / 2
    assert abs(alpha_unmix(mid, ca, cb) - 0.5) < 1e-9


def test_vectorized_and_clipped():
    ca = np.array([0.0, 0.0, 0.0]); cb = np.array([100.0, 0.0, 0.0])
    V = np.array([[-50.0, 0, 0], [50.0, 0, 0], [150.0, 0, 0]])  # below, mid, above
    a = alpha_unmix(V, ca, cb)
    assert a.shape == (3,)
    assert a[0] == 1.0 and abs(a[1] - 0.5) < 1e-9 and a[2] == 0.0  # clipped to [0,1]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_softlabel.py -k unmix -v`
Expected: FAIL — `vectormark.softlabel` does not exist.

- [ ] **Step 3: Implement**

Create `src/vectormark/softlabel.py`:

```python
# SPDX-License-Identifier: MIT
"""Antialiasing-aware soft label field: recover per-region sub-pixel coverage from the
pre-quantization RGB + palette, so contour extraction is smooth and seams are shared.
Pure numpy; no pipeline imports."""

from __future__ import annotations

import numpy as np


def alpha_unmix(rgb: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> np.ndarray:
    """Coverage of color A in a two-color blend V = α·c_a + (1−α)·c_b.
    α = clip((V−c_b)·(c_a−c_b)/|c_a−c_b|², 0, 1). α=1 ⇒ pure c_a. Vectorized over leading
    dims of `rgb` (last axis = channels)."""
    rgb = np.asarray(rgb, float); c_a = np.asarray(c_a, float); c_b = np.asarray(c_b, float)
    d = c_a - c_b
    denom = float(d @ d) or 1.0
    return np.clip(((rgb - c_b) @ d) / denom, 0.0, 1.0)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_softlabel.py -k unmix -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/softlabel.py tests/test_softlabel.py
git commit -m "feat(softlabel): alpha_unmix — coverage from a two-color blend

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: `soft_label_field` — global partition-of-unity membership

**Files:**
- Modify: `src/vectormark/softlabel.py`
- Test: `tests/test_softlabel.py`

**Interfaces:**
- Consumes: `alpha_unmix` (Task 2); `color.srgb_to_oklab` (for ΔE).
- Produces: `soft_label_field(rgb, palette) -> np.ndarray` — `rgb` is `(H,W,3)` float (pre-quantization, 0-255), `palette` is `(K,3)` uint8/float (INCLUDING the background color as a row). Returns `L` of shape `(H,W,K)`, partition-of-unity (`L.sum(axis=2) ≈ 1`), where `L[...,k]` is pixel membership in palette color k. Construction by band: interior pixels (far from any label change) are one-hot for their nearest palette color (anchors thin features); two-color boundary band uses `alpha_unmix` of the locally-dominant pair; ≥3-color junction band uses normalized inverse-ΔE membership (value-tiebroken). Deterministic.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_softlabel.py`:

```python
from vectormark.softlabel import soft_label_field


def _two_color_ramp(H=40, W=60):
    # left half color A, right half color B, with a 2px antialiased ramp at x=W/2
    A = np.array([20.0, 40.0, 200.0]); B = np.array([255.0, 255.0, 255.0])
    img = np.empty((H, W, 3))
    for x in range(W):
        t = np.clip((x - (W / 2 - 1)) / 2.0, 0, 1)   # 0 left of seam, 1 right, ramp across 2px
        img[:, x] = (1 - t) * A + t * B
    return img, np.array([A, B], np.uint8)


def test_partition_of_unity():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    assert L.shape == (40, 60, 2)
    assert np.allclose(L.sum(axis=2), 1.0, atol=1e-6)        # memberships sum to 1


def test_interior_is_one_hot():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    # far-left column is pure A -> L[...,0] == 1 (interior anchoring)
    assert np.allclose(L[:, 0, 0], 1.0) and np.allclose(L[:, 0, 1], 0.0)
    assert np.allclose(L[:, -1, 0], 0.0) and np.allclose(L[:, -1, 1], 1.0)


def test_seam_band_crosses_half():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    # along a row, A's membership decreases monotonically across the seam and passes 0.5
    row = L[20, :, 0]
    assert row[0] > 0.9 and row[-1] < 0.1
    assert np.any(np.abs(row - 0.5) < 0.1)                   # a 0.5 crossing exists
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_softlabel.py -k "partition or one_hot or seam_band" -v`
Expected: FAIL — `soft_label_field` not defined.

- [ ] **Step 3: Implement**

Add to `src/vectormark/softlabel.py` (import `from .color import srgb_to_oklab` at top):

```python
def _oklab_dist(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H,W,K) OKLab distance from each pixel to each palette color."""
    px = srgb_to_oklab(rgb / 255.0)                        # (H,W,3)
    pal = srgb_to_oklab(np.asarray(palette, float) / 255.0)  # (K,3)
    return np.linalg.norm(px[:, :, None, :] - pal[None, None, :, :], axis=3)


def soft_label_field(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Per-pixel partition-of-unity membership in each palette color (background included
    as a row). Interior one-hot (anchors thin features); two-color band alpha-unmixed;
    >=3-color junction band normalized-inverse-ΔE. Deterministic (value-ordered ties)."""
    rgb = np.asarray(rgb, float)
    palette = np.asarray(palette, float)
    H, W, _ = rgb.shape
    K = len(palette)
    dist = _oklab_dist(rgb, palette)                       # (H,W,K)
    # rank the two nearest labels per pixel; value-ordered ties via a tiny index bias
    order = np.argsort(dist + np.arange(K) * 1e-12, axis=2)  # (H,W,K) ascending
    n0 = order[..., 0]; n1 = order[..., 1]
    d0 = np.take_along_axis(dist, n0[..., None], 2)[..., 0]
    d1 = np.take_along_axis(dist, n1[..., None], 2)[..., 0]

    L = np.zeros((H, W, K), float)
    # interior: clearly one color (nearest is much closer than runner-up) -> one-hot
    interior = d0 < 0.5 * d1                                # nearest dominates
    np.put_along_axis(L, n0[..., None], np.where(interior, 1.0, 0.0)[..., None], 2)

    # boundary band (not interior): unmix the two locally-dominant colors
    band = ~interior
    if band.any():
        by, bx = np.where(band)
        ca = palette[n0[by, bx]]; cb = palette[n1[by, bx]]
        a = np.empty(len(by))
        for i in range(len(by)):
            a[i] = alpha_unmix(rgb[by[i], bx[i]], ca[i], cb[i])
        L[by, bx, n0[by, bx]] = a
        L[by, bx, n1[by, bx]] = 1.0 - a

    # junction band: a pixel with >=3 comparably-near labels is ill-posed for unmix;
    # detect (3rd-nearest within 1.3x of nearest) and overwrite with inverse-ΔE membership.
    d2 = np.take_along_axis(dist, order[..., 2][..., None], 2)[..., 0] if K >= 3 else np.full((H, W), np.inf)
    junction = band & (d2 < 1.3 * np.maximum(d0, 1e-9))
    if junction.any():
        inv = 1.0 / (dist[junction] + 1e-6)
        L[junction] = inv / inv.sum(axis=1, keepdims=True)

    return L
```

> Note for the implementer: the per-pixel Python loop in the band is for clarity; if profiling shows it hot on real images, vectorize it — `alpha_unmix` already accepts arrays, so gather `ca`/`cb` as `(M,3)` and compute `a` in one call. Keep the result identical. The interior threshold `d0 < 0.5*d1` and junction `d2 < 1.3*d0` are starting points calibrated in Task 6.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_softlabel.py -k "partition or one_hot or seam_band" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/softlabel.py tests/test_softlabel.py
git commit -m "feat(softlabel): soft_label_field — partition-of-unity membership by band

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: `region_coverage` — signed margin with the gap-free guarantee

**Files:**
- Modify: `src/vectormark/softlabel.py`
- Test: `tests/test_softlabel.py`

**Interfaces:**
- Consumes: `soft_label_field` (Task 3).
- Produces: `region_coverage(L, k, region_mask) -> np.ndarray` — given the global field `L` (H,W,K), a color index `k`, and the region's bool mask (one connected component of color k), returns `cov` (H,W) float = `(φ_k + 1)/2` where `φ_k = L[...,k] − max_{j≠k} L[...,j]`, then zeroed outside a 2px dilation of `region_mask` (so only THIS component's boundary is traced, while the seam band around it is preserved). Boundary sits at 0.5. Because every region's cov is derived from the SAME `L`, an A|B seam has `φ_B = −φ_A` exactly → shared seams are point-identical.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_softlabel.py`:

```python
import numpy as np
from scipy import ndimage
from vectormark.softlabel import soft_label_field, region_coverage
from vectormark.contour import outer_contour


def test_region_coverage_boundary_at_half():
    img, pal = _two_color_ramp(H=40, W=60)
    L = soft_label_field(img, pal)
    mask_a = np.zeros((40, 60), bool); mask_a[:, :30] = True
    cov = region_coverage(L, 0, mask_a)
    assert np.all(cov[:, 0] > 0.9) and np.all(cov[:, -1] < 0.1)   # 1 inside A, 0 outside


def test_shared_seam_is_point_identical():
    # THE SPINE: region A (color 0) and region B (color 1) share the seam; the sub-arcs
    # along the seam must be identical (φ_B = −φ_A) → no gap, no overlap.
    img, pal = _two_color_ramp(H=40, W=60)
    L = soft_label_field(img, pal)
    mask_a = np.zeros((40, 60), bool); mask_a[:, :30] = True
    mask_b = np.zeros((40, 60), bool); mask_b[:, 30:] = True
    cov_a = region_coverage(L, 0, mask_a)
    cov_b = region_coverage(L, 1, mask_b)
    # along the shared seam column band, cov_b == 1 - cov_a (φ_B = −φ_A exactly)
    seam = slice(28, 32)
    assert np.allclose(cov_b[:, seam], 1.0 - cov_a[:, seam], atol=1e-9)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_softlabel.py -k "region_coverage or shared_seam" -v`
Expected: FAIL — `region_coverage` not defined.

- [ ] **Step 3: Implement**

Add to `src/vectormark/softlabel.py` (import `from scipy import ndimage`):

```python
def region_coverage(L: np.ndarray, k: int, region_mask: np.ndarray, *, reach: int = 2) -> np.ndarray:
    """Region k's coverage field from the shared global L: cov = (φ+1)/2 with
    φ = L[...,k] − max_{j≠k} L[...,j] (boundary at 0.5), zeroed outside a `reach`-px
    dilation of `region_mask` so only this component's boundary is traced. Derived from
    the SAME L for every region ⇒ φ_B = −φ_A on shared seams (gap-free)."""
    others = np.delete(L, k, axis=2).max(axis=2)
    phi = L[..., k] - others
    cov = (phi + 1.0) / 2.0
    near = ndimage.binary_dilation(region_mask, iterations=reach)
    cov = np.where(near, cov, 0.0)
    return cov
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_softlabel.py -k "region_coverage or shared_seam" -v`
Expected: PASS — the shared-seam test confirms the no-gap identity numerically.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/softlabel.py tests/test_softlabel.py
git commit -m "feat(softlabel): region_coverage — signed margin, gap-free shared seams

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Wire the coverage field into the pipeline

**Files:**
- Modify: `src/vectormark/pipeline.py` (`_segment_image`)
- Modify: `src/vectormark/contour.py` (`significant_contours`, `region_corner_radius` — thread `coverage`)
- Modify: `src/vectormark/selector.py` (`generate_geometry_candidates` contour call), `src/vectormark/occlusion.py` (the two `region_contours` calls)
- Test: `tests/test_aa_contour_acceptance.py`

**Interfaces:**
- Consumes: `soft_label_field`, `region_coverage` (Tasks 3-4); `segment._background_color`, `color.extract_palette`/`quantize`.
- Produces: `_segment_image` attaches `region.coverage` to every returned region; the contour call sites pass `region.coverage`. With the field attached, `idealize` extracts smooth contours; nothing downstream of segmentation changes shape.

**Context:** `_segment_image` (pipeline.py:60-67) currently does `palette = extract_palette(...)`, `q = quantize(arr, palette)`, `regions = segment(q, ...)`. After segmentation, build the field once and attach coverage per region. The region's color index in the field is found by matching its `color_hex` to a palette row. The background color (from `segment._background_color(q)`) must be appended to the palette passed to `soft_label_field` so it competes as a label.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_aa_contour_acceptance.py`:

```python
import numpy as np
from vectormark.pipeline import _segment_image, Options


def _two_blob_img(H=80, W=120):
    img = np.full((H, W, 3), 255, np.uint8)              # white background
    img[20:60, 15:55] = (20, 40, 200)                    # blue block (AA-free synthetic)
    img[20:60, 65:105] = (220, 60, 20)                   # orange block
    return img


def test_segment_attaches_coverage():
    w, h, regions = _segment_image(_two_blob_img(), Options(max_colors=16))
    assert regions, "expected regions"
    for r in regions:
        assert r.coverage is not None
        assert r.coverage.shape == r.mask.shape
        # coverage is ~1 on the region interior, present where the mask is
        assert r.coverage[r.mask].mean() > 0.8


def test_idealize_still_runs_with_coverage():
    from vectormark.pipeline import idealize
    svg = idealize(_two_blob_img(), options=Options(max_colors=16))
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_aa_contour_acceptance.py -k "attaches or still_runs" -v`
Expected: FAIL — `_segment_image` does not attach `coverage` yet.

- [ ] **Step 3: Implement the wiring**

In `src/vectormark/pipeline.py`, extend `_segment_image` (after `regions = segment(q, min_area=16)` and the size cut, before `return`):

```python
    from .softlabel import soft_label_field, region_coverage
    from .segment import _background_color, hexstr
    if regions:
        bg = _background_color(q)
        # palette as the unique region colors + background, in a stable order; index by hex
        colors = list({r.color_hex: None for r in regions}.keys())
        rows = [tuple(int(c) for c in _hex_to_rgb(hx)) for hx in colors] + [bg]
        palette = np.array(rows, np.uint8)
        hex_to_idx = {hx: i for i, hx in enumerate(colors)}
        L = soft_label_field(arr.astype(float), palette)
        for r in regions:
            r.coverage = region_coverage(L, hex_to_idx[r.color_hex], r.mask)
```

Add a small `_hex_to_rgb(hx: str) -> tuple[int,int,int]` helper in `pipeline.py` (or import `hexstr`'s inverse if one exists — check `segment.py`/`color.py` first with `rg "hex_to_rgb|def hexstr"`; reuse if present, else: `return (int(hx[1:3],16), int(hx[3:5],16), int(hx[5:7],16))`).

In `src/vectormark/contour.py`, thread `coverage` through `significant_contours` and `region_corner_radius`:
- `significant_contours(mask, *, min_hole_fraction=HOLE_AREA_FRACTION, coverage=None)` → pass `coverage` to its internal `region_contours(mask, coverage=coverage)` call.
- `region_corner_radius(mask, *, coverage=None)` → pass `coverage` to its `region_contours(mask, coverage=coverage)` call (contour.py:177).

In `src/vectormark/selector.py`, change `significant_contours(region.mask)` (selector.py:75) → `significant_contours(region.mask, coverage=region.coverage)`. In `pipeline.py`'s `build_candidates`, change `region_corner_radius(region.mask)` / `region_corner_radius(footprint.mask)` calls to pass `coverage=region.coverage` (footprints have no coverage → `coverage=None`, fine). In `src/vectormark/occlusion.py`, change both `region_contours(region.mask)` calls (occlusion.py:45, :236) → `region_contours(region.mask, coverage=region.coverage)`.

- [ ] **Step 4: Run the integration test + full suite**

Run: `uv run pytest tests/test_aa_contour_acceptance.py -k "attaches or still_runs" -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: some geometry/golden tests MAY shift because contours are now smoother (better fits, fewer NOFIT). Re-derive each changed golden from the new (smoother) output; do NOT weaken — list every changed test in the report/commit with a reason. The pre-existing MCP-stdio failure is not yours.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/pipeline.py src/vectormark/contour.py src/vectormark/selector.py src/vectormark/occlusion.py tests/test_aa_contour_acceptance.py
git commit -m "feat(pipeline): attach soft-coverage field to regions; extract sub-pixel contours

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Acceptance — smoothness, junction, thin-feature, determinism, V-bird, corpus

**Files:**
- Test: `tests/test_aa_contour_acceptance.py`
- Modify: calibration constants in `softlabel.py` only if the corpus/junction tests show it

**Interfaces:**
- Consumes: the full AA-contour pipeline (Tasks 1-5).
- Context: V-bird at `/Users/pmouli/GitHub.nosync/active/py/vectormark/scratch/real-logos/vbird.png`; corpus at `scratch/real-logos/` (UNTRACKED — never `git add scratch/`).

- [ ] **Step 1: Write the acceptance tests (the spec's test list)**

Add to `tests/test_aa_contour_acceptance.py`:

```python
def test_smoothness_beats_binary_on_aa_circle():
    # an antialiased circle: soft-field contour RMS to the true circle is tighter than binary
    H = 120; yy, xx = np.ogrid[:H, :H]
    d = np.sqrt((yy - 60.0) ** 2 + (xx - 60.0) ** 2)
    cov = np.clip(0.5 + (40.0 - d), 0, 1); mask = d <= 40
    from vectormark.contour import outer_contour
    def rms(c): return float(np.sqrt(np.mean((np.hypot(c[:,0]-60.0, c[:,1]-60.0) - 40.0)**2)))
    assert rms(outer_contour(mask, coverage=cov)) < 0.15
    assert rms(outer_contour(mask, coverage=cov)) < rms(outer_contour(mask))


def test_determinism_bit_identical():
    img = _two_blob_img()
    w1, h1, r1 = _segment_image(img, Options(max_colors=16))
    w2, h2, r2 = _segment_image(img, Options(max_colors=16))
    for a, b in zip(r1, r2):
        assert np.array_equal(a.coverage, b.coverage)


def test_thin_feature_survives():
    # a 1px-ish AA stroke must not collapse (interior one-hot anchoring)
    img = np.full((40, 60, 3), 255, np.uint8); img[:, 29:31] = (20, 40, 200)
    _, _, regions = _segment_image(img, Options(max_colors=16))
    assert any(r.coverage[r.mask].mean() > 0.5 for r in regions)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_aa_contour_acceptance.py -v`
Expected: PASS (smoothness, determinism, thin-feature). If `test_thin_feature_survives` fails, the interior-anchoring threshold (`d0 < 0.5*d1` in `soft_label_field`) is too aggressive — loosen it and re-run; do NOT remove anchoring.

- [ ] **Step 3: Junction + gap-free integration check**

Add a triple-point test (Mercedes-star / three-wedge image): three colors meet at a center; extract all three regions' contours and assert no gap/overlap sliver above tolerance at the junction (supersample-rasterize the three filled coverage fields; every pixel covered by exactly one region within the junction neighborhood, tolerance ≤ 1 sliver pixel). If this FAILS, that is the documented trigger for Strategy 2 (planar seam graph) — STOP and report it to the controller rather than building Strategy 2 unprompted.

- [ ] **Step 4: Corpus regression + V-bird (run, don't over-tune)**

Render the corpus through the AA pipeline (background harness, the corpus is slow) and confirm: no crashes, no malformed SVG, and — the win — the V-bird's dots come out as `<circle>` (the smooth contour now passes the robust-RMS gate) and fewer logos hit `NOFIT` than the grammar-only baseline (compare `idealize(report=True)` strategy histograms). Record the before/after NOFIT counts. Do NOT tune the bounded-grammar RMS threshold here (separate follow-up); only calibrate `softlabel.py`'s band thresholds if a genuine break (crash/all-collapse) appears.

- [ ] **Step 5: Full suite + commit**

Run: `uv run pytest -q`
Expected: green.

```bash
git add tests/test_aa_contour_acceptance.py src/vectormark/softlabel.py
git commit -m "test(aa-contour): smoothness/junction/thin/determinism + V-bird acceptance

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: `uv run pytest -q` green; render the V-bird and confirm the dots are now smooth circles and the wing seams are gap-free + smooth (vs the grammar-only baseline). Dispatch a whole-branch review (most-capable model) over `master..HEAD` — pay special attention to the gap-free guarantee (the `region_coverage` negation identity) and that no bool-`mask` consumer was disturbed. Then superpowers:finishing-a-development-branch to open the PR; note any junction-sliver follow-up (Strategy 2). PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
