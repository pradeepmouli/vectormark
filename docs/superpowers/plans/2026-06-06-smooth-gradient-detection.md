# Smooth-Gradient Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect a smooth (non-posterized) gradient that fills a logo's mark and emit it as one gradient shape, so real app-icon gradient logos idealize to a gradient instead of a flat colour.

**Architecture:** Add a "smooth-silhouette" path inside `detect_gradients` that runs *after* the existing band-grouping loop on the leftover regions: if the leftover foreground is one dominant connected blob (≥85%) and `fit_gradient` accepts a model fit to its raw pixels, emit one gradient over that silhouette. Reuses the existing fit + ΔE gate + emit machinery — **no pipeline changes**.

**Tech Stack:** Python, numpy, scipy.ndimage (already imported in `gradient.py` as `ndi`), pytest, `uv`.

Spec: `docs/superpowers/specs/2026-06-06-smooth-gradient-detection-design.md`. Branch: `feat/gradients-smooth` (stacked on `feat/gradients`).

---

## File Structure

- **`src/vectormark/gradient.py`** (modify): add constant `_BLOB_DOMINANCE`, helper `_dominant_blob_fraction`, and the smooth-silhouette branch in `detect_gradients`. This is the only source file touched.
- **`tests/test_gradient.py`** (append): unit tests for `_dominant_blob_fraction` and the new `detect_gradients` branch.
- **`tests/test_acceptance_smooth_gradient.py`** (create): end-to-end tests through `idealize` proving the smooth path emits a gradient and rejects flats/multi-blob.

Current `detect_gradients` (for reference — do not retype the band-grouping loop, only append the smooth branch):

```python
def detect_gradients(
    regions: list[Region], rgb_image: np.ndarray
) -> tuple[list[tuple[Region, dict]], list[Region]]:
    """Group ramp bands, fit+gate a gradient per footprint, and return
    (accepted [(footprint_region, model)], remaining flat regions)."""
    fills: list[tuple[Region, dict]] = []
    consumed: set[int] = set()
    for group in _ramp_groups(regions):
        mask = np.zeros(rgb_image.shape[:2], bool)
        for m in group:
            mask |= m.mask
        model = fit_gradient(mask, rgb_image)
        if model is None:
            continue                                   # dissolve back into flat bands
        mask = _expand_footprint(model, mask, rgb_image)
        rep = max(group, key=lambda r: r.area)
        footprint = Region(label=rep.label, mask=mask, color_hex=rep.color_hex)
        fills.append((footprint, model))
        consumed.update(m.label for m in group)
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
```

---

### Task 1: `_dominant_blob_fraction` helper + constant

**Files:**
- Modify: `src/vectormark/gradient.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gradient.py`:

```python
def test_dominant_blob_fraction():
    from vectormark.gradient import _dominant_blob_fraction
    m = np.zeros((20, 40), bool)
    m[2:18, 2:18] = True                       # one 16x16 blob, rest empty
    assert _dominant_blob_fraction(m) == 1.0
    m[2:18, 22:38] = True                      # add a second, equal, disconnected blob
    assert abs(_dominant_blob_fraction(m) - 0.5) < 1e-9
    assert _dominant_blob_fraction(np.zeros((5, 5), bool)) == 0.0   # empty -> 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gradient.py -k dominant_blob -q`
Expected: FAIL — `_dominant_blob_fraction` not defined.

- [ ] **Step 3: Add the constant and helper**

In `src/vectormark/gradient.py`, add the constant next to the other module constants (right after the `_GATE_DELTA_E = ...` line near the top):

```python
_BLOB_DOMINANCE = 0.85   # smooth-gradient path: min fraction of the foreground that must lie
                         # in a single connected component for the mark to be treated as one
                         # gradient blob (rejects multi-glyph wordmarks before any fit).
```

Add the helper immediately above `def detect_gradients(` (after `_expand_footprint`):

```python
def _dominant_blob_fraction(mask: np.ndarray) -> float:
    """Fraction of the foreground occupied by its largest connected component
    (4-connectivity). 1.0 = one solid blob; ~0 = many disconnected pieces; 0.0 if empty."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    labels, n = ndi.label(mask)
    if n == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]       # drop background label 0
    return float(sizes.max()) / total
```

(`ndi` is already imported as `from scipy import ndimage as ndi`; `np` is already imported.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_gradient.py -k dominant_blob -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): _dominant_blob_fraction helper + _BLOB_DOMINANCE"
```

Commit trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

---

### Task 2: smooth-silhouette branch in `detect_gradients`

**Files:**
- Modify: `src/vectormark/gradient.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradient.py`:

```python
def _smooth_linear_region(h, w, c0, c1):
    """One full-canvas Region + a horizontally smooth linear gradient image (raw pixels)."""
    from vectormark.types import Region
    yy, xx = np.mgrid[:h, :w]
    t = xx / (w - 1)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = c0[ch] + t * (c1[ch] - c0[ch])
    img = img.round().astype(np.uint8)
    return [Region(label=1, mask=np.ones((h, w), bool), color_hex="#000000")], img


def test_detect_gradients_smooth_single_blob_fits():
    from vectormark.gradient import detect_gradients
    regions, img = _smooth_linear_region(60, 120, (20, 40, 200), (220, 40, 90))
    fills, remaining = detect_gradients(regions, img)
    assert len(fills) == 1 and fills[0][1]["kind"] == "linear"
    assert remaining == []                                   # the blob was consumed


def test_detect_gradients_smooth_rejects_multiblob():
    from vectormark.types import Region
    from vectormark.gradient import detect_gradients
    h, w = 60, 140
    img = np.full((h, w, 3), 255, np.uint8)
    img[10:50, 10:50] = (220, 30, 30)                        # two disconnected flat blocks
    img[10:50, 90:130] = (30, 30, 220)
    m1 = np.zeros((h, w), bool); m1[10:50, 10:50] = True
    m2 = np.zeros((h, w), bool); m2[10:50, 90:130] = True
    regions = [Region(label=1, mask=m1, color_hex="#dc1e1e"),
               Region(label=2, mask=m2, color_hex="#1e1edc")]
    fills, remaining = detect_gradients(regions, img)
    assert fills == [] and {r.label for r in remaining} == {1, 2}   # dom 0.5 < 0.85


def test_detect_gradients_smooth_rejects_flat_blob():
    from vectormark.types import Region
    from vectormark.gradient import detect_gradients
    h, w = 50, 50
    img = np.full((h, w, 3), (40, 120, 200), np.uint8)       # one flat colour
    regions = [Region(label=1, mask=np.ones((h, w), bool), color_hex="#2878c8")]
    fills, remaining = detect_gradients(regions, img)
    assert fills == [] and {r.label for r in remaining} == {1}   # fit_gradient -> None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k "smooth_single_blob or smooth_rejects" -q`
Expected: FAIL — `test_detect_gradients_smooth_single_blob_fits` fails (no smooth path yet, so `fills == []`). The two `rejects` tests may already pass (no fill is produced today); that is fine — they lock the negative behaviour after the change.

- [ ] **Step 3: Add the smooth-silhouette branch**

In `src/vectormark/gradient.py`, in `detect_gradients`, replace the final two lines:

```python
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
```

with:

```python
    # smooth-gradient path: band-grouping only fires on posterized ramps (≥3 adjacent bands).
    # A smooth ramp collapses to ~1 region at palette extraction, so test the leftover mark as
    # a single gradient blob fit to the ORIGINAL pixels (see the design spec). Guard with a
    # dominant-connected-blob check so multi-glyph wordmarks can't be fit as one gradient.
    leftover = [r for r in regions if r.label not in consumed]
    if leftover:
        sil = np.zeros(rgb_image.shape[:2], bool)
        for r in leftover:
            sil |= r.mask
        if _dominant_blob_fraction(sil) >= _BLOB_DOMINANCE:
            model = fit_gradient(sil, rgb_image)
            if model is not None:
                rep = max(leftover, key=lambda r: r.area)
                fills.append((Region(label=rep.label, mask=sil, color_hex=rep.color_hex), model))
                consumed.update(r.label for r in leftover)
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k "smooth_single_blob or smooth_rejects" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole gradient unit suite (regression check)**

Run: `uv run pytest tests/test_gradient.py -q`
Expected: all green. The smooth branch is additive; verify in particular that `test_detect_gradients_consumes_ramp_returns_remaining` (leftover flat green → `fit_gradient` None → no smooth fill) and `test_detect_gradients_dissolves_unfittable_group_back_to_flats` (zig-zag silhouette → `fit_gradient` None → no smooth fill) still pass.

- [ ] **Step 6: Commit**

```bash
git add src/vectormark/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): smooth-silhouette gradient path in detect_gradients"
```

Commit trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

---

### Task 3: end-to-end acceptance + regression

**Files:**
- Create: `tests/test_acceptance_smooth_gradient.py`

End-to-end through `idealize`. Each gradient test asserts (a) band-grouping does **not** fire on the segmented image (so the gradient can *only* come from the new smooth path) and (b) one gradient def is emitted and the render matches within the ΔE bar.

- [ ] **Step 1: Write the acceptance tests**

Create `tests/test_acceptance_smooth_gradient.py`:

```python
"""Smooth-gradient reconstruction end-to-end through idealize. A smooth (non-posterized)
gradient mark becomes one shape + one <linearGradient>/<radialGradient>; flats and multi-blob
marks stay flat. Each positive test first asserts band-grouping finds nothing, so the gradient
is attributable to the smooth-silhouette path, not the band-grouping path."""

import numpy as np

from vectormark import Options, idealize
from vectormark.color import mean_delta_e
from vectormark.gradient import _ramp_groups
from vectormark.pipeline import _segment_image
from tests._render import render_svg


def _smooth_linear_rect(h, w, x0, x1, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    t = ((xx - x0) / (x1 - x0)).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    rect = np.zeros((h, w), bool); rect[int(h * 0.2):int(h * 0.8), x0:x1] = True
    for ch in range(3):
        img[:, :, ch][rect] = (c0[ch] + t * (c1[ch] - c0[ch]))[rect]
    return img.round().astype(np.uint8)


def _smooth_radial_disc(h, w, c, r, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - c[0], yy - c[1])
    t = (dist / r).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    disc = dist <= r
    for ch in range(3):
        img[:, :, ch][disc] = (c0[ch] + t * (c1[ch] - c0[ch]))[disc]
    return img.round().astype(np.uint8)


def test_smooth_linear_rect_via_smooth_path():
    h, w = 160, 240
    # low-contrast blue ramp: smooth enough that palette extraction collapses it (no bands)
    img = _smooth_linear_rect(h, w, 40, 200, (90, 150, 230), (60, 110, 205))
    sw, sh, regions = _segment_image(img, Options())
    assert _ramp_groups(regions) == []                  # band-grouping does NOT fire
    svg = idealize(img, options=Options())
    assert svg.count("<linearGradient") == 1            # ...so the smooth path produced it
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.06


def test_smooth_radial_disc_via_smooth_path():
    h, w = 200, 200
    img = _smooth_radial_disc(h, w, (100, 100), 85, (120, 190, 245), (70, 120, 210))
    sw, sh, regions = _segment_image(img, Options())
    assert _ramp_groups(regions) == []                  # band-grouping does NOT fire
    svg = idealize(img, options=Options())
    assert svg.count("<radialGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.07


def test_smooth_two_blob_stays_flat():
    h, w = 160, 260
    img = np.full((h, w, 3), 255, np.uint8)
    g = np.linspace(0.0, 1.0, 80)
    for x0 in (20, 160):                                 # two disconnected smooth squares
        block = np.empty((80, 80, 3))
        for ch, (a, b) in enumerate(zip((90, 150, 230), (60, 110, 205))):
            block[:, :, ch] = a + g[None, :] * (b - a)
        img[40:120, x0:x0 + 80] = block.round().astype(np.uint8)
    svg = idealize(img, options=Options())
    assert "<linearGradient" not in svg and "<radialGradient" not in svg   # dom 0.5 < 0.85
```

- [ ] **Step 2: Run the acceptance tests**

Run: `uv run pytest tests/test_acceptance_smooth_gradient.py -q`
Expected: PASS (3 passed).

Tuning latitude (only if a test fails for the reason given):
- If a positive test's `assert _ramp_groups(regions) == []` FAILS (band-grouping fired because the synthetic gradient was contrasty enough to posterize into ≥3 bands), **lower the contrast** — move `c0`/`c1` closer together — until band-grouping finds nothing while the endpoints still differ enough that a gradient is emitted. Do NOT delete the assertion; its whole purpose is to prove the smooth path.
- If a positive test emits the gradient but the render-ΔE just exceeds the bar, the fit is firing but imperfect: do NOT loosen the bar beyond 0.06 (linear) / 0.07 (radial). Report it.
- If `test_smooth_two_blob_stays_flat` emits a gradient (it must not), the dominant-blob guard or its threshold is wrong — report rather than loosening the test.

- [ ] **Step 3: Full regression**

Run: `uv run pytest -q`
Expected: all green. Prior suite was 121; this plan adds 1 (Task 1) + 3 (Task 2) + 3 (Task 3) = **128 passed**. Confirm no existing test regressed (the smooth branch is additive and gated; PR #8's `test_two_color_nonramp_stays_flat` and `test_flat_logo_not_gradientified` must still hold).

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance_smooth_gradient.py
git commit -m "test(acceptance): smooth-gradient reconstruction (linear, radial) + multi-blob negative"
```

Commit trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

---

## Manual eval (not a committed test)

After Task 3, re-run the real-logo eval to confirm Telegram and Apple Music now emit gradients end-to-end (they are untracked brand assets in `scratch/real-logos/`, kept out of the suite for licensing):

```bash
uv run python /tmp/realeval.py   # or re-create: composite-on-white -> idealize -> count defs + ΔE
```
Expected: `telegram` and `apple_music` now show `lin=1` (or `rad=1`) with low ΔE; `slack`, `microsoft`, `gdrive`, `sketch`, `asana`, `instagram` remain gradient-free.

---

## Self-Review

**1. Spec coverage:**
- Whole-silhouette raw-pixel fit + gate + dominant-blob guard → Task 2 (`detect_gradients` branch) + Task 1 (`_dominant_blob_fraction`, `_BLOB_DOMINANCE`). ✓
- Runs after band-grouping on leftover regions; no pipeline changes → Task 2 (branch placement, emits via existing `fills`). ✓
- Reuse `fit_gradient` / `_agreement_delta_e` / existing emit; no `_expand_footprint` on this path → Task 2. ✓
- Edge handling (no regions / multi-blob / fit None) → Task 2 tests (`rejects_multiblob`, `rejects_flat_blob`). ✓
- Synthetic fixtures, real logos untracked → Task 3 + Manual eval. ✓
- Determinism (`ndi.label`) → Task 1. ✓
- Documented limitation (multi-blob gradient+element) → covered by `test_smooth_two_blob_stays_flat` asserting the safe miss. ✓

**2. Placeholder scan:** No TBD/vague steps; every code step is complete; tuning notes name concrete bounds (contrast lever, ΔE bars 0.06/0.07). ✓

**3. Type consistency:** `_dominant_blob_fraction(mask) -> float`, `_BLOB_DOMINANCE` (float), `detect_gradients(regions, rgb_image) -> (list[(Region, dict)], list[Region])` (unchanged signature), `fit_gradient(mask, rgb_image) -> dict | None`, model dict `{"kind","geometry","stops"}` — all consistent with PR #8 and across tasks. ✓
