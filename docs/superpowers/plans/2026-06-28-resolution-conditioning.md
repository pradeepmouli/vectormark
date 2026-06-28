# Resolution Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Normalize oversized inputs to a working resolution before segmentation (downscale only, no denoise), preserving original output dimensions.

**Architecture:** A `_condition_input` helper downscales the flattened RGB array when its longest side exceeds `Options.working_max_dim` (LANCZOS, aspect-preserving). `idealize` conditions at the top and rewrites the emitted SVG `width`/`height` back to the original pixel size (viewBox stays in working space — a pure display scale).

**Tech Stack:** numpy, Pillow (`Image.resize(..., Image.LANCZOS)`).

## Global Constraints

- Python ≥ 3.12, pure-Python. TDD. `rg` not `grep`. Determinism: LANCZOS is deterministic, no RNG.
- Conditioning is **downscale only** — never upscale, never denoise/blur.
- Default-on: `Options.working_max_dim: int | None = 768`; `None` disables (pure pass-through).
- Do NOT `git add scratch/`. Commit trailer EXACTLY: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Pre-existing `test_stdio_server_exposes_idealize_logo_tool` (MCP) failure, if present, is not yours.

---

### Task 1: `working_max_dim` option + `_condition_input` helper

**Files:**
- Modify: `src/vectormark/pipeline.py` (add `working_max_dim` to `Options`; add `_condition_input`)
- Test: `tests/test_conditioning.py` (create)

**Interfaces:**
- Produces: `_condition_input(arr: np.ndarray, working_max_dim: int | None) -> np.ndarray` — returns `arr` unchanged when `working_max_dim` is None or `max(H,W) <= working_max_dim`; else a LANCZOS-downscaled `(H',W',3) uint8` whose longest side equals `working_max_dim`, aspect-preserving (`round`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from vectormark.pipeline import _condition_input, Options

def test_passthrough_when_within_threshold():
    arr = np.zeros((400, 300, 3), np.uint8)
    out = _condition_input(arr, 768)
    assert out is arr or (out.shape == arr.shape and np.array_equal(out, arr))

def test_passthrough_when_disabled():
    arr = np.zeros((2000, 1000, 3), np.uint8)
    assert np.array_equal(_condition_input(arr, None), arr)

def test_downscales_longest_side_to_target():
    arr = np.zeros((1500, 900, 3), np.uint8)
    out = _condition_input(arr, 768)
    assert max(out.shape[:2]) == 768
    assert out.shape[:2] == (768, round(900 * 768 / 1500))   # aspect preserved
    assert out.dtype == np.uint8

def test_deterministic():
    rng = np.zeros((1200, 1200, 3), np.uint8); rng[300:900, 300:900] = (20, 40, 200)
    assert np.array_equal(_condition_input(rng, 512), _condition_input(rng, 512))

def test_default_option_is_768():
    assert Options().working_max_dim == 768
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_conditioning.py -v`
Expected: FAIL (`_condition_input` / `working_max_dim` undefined).

- [ ] **Step 3: Implement**

Add to `Options` (after `fidelity_tol` / `selection`):
```python
    working_max_dim: int | None = 768  # downscale inputs whose longest side exceeds this
                                        # (LANCZOS) before segmentation; None disables.
```

Add the helper (near `_flatten_on_white`):
```python
def _condition_input(arr: np.ndarray, working_max_dim: int | None) -> np.ndarray:
    """Downscale an oversized RGB array to a working resolution before segmentation, so
    input noise stops fragmenting at high pixel counts. Longest side -> working_max_dim,
    aspect-preserving, LANCZOS. Returns arr unchanged when disabled or already small.
    Downscale only — never upscale, never denoise."""
    if working_max_dim is None:
        return arr
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= working_max_dim:
        return arr
    scale = working_max_dim / longest
    new_w, new_h = round(w * scale), round(h * scale)
    img = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_conditioning.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/pipeline.py tests/test_conditioning.py
git commit -m "feat(pipeline): _condition_input + working_max_dim option (downscale oversized inputs)"
```

---

### Task 2: Wire conditioning into `idealize`; preserve original output size

**Files:**
- Modify: `src/vectormark/pipeline.py` (`idealize`; add `_set_svg_output_size`)
- Test: `tests/test_conditioning.py` (extend)

**Interfaces:**
- Consumes: `_condition_input` (Task 1).
- Produces: `idealize` conditions the array before `_segment_image`; the returned SVG's `width`/`height` equal the ORIGINAL input pixel size while `viewBox` is in working space.

- [ ] **Step 1: Write the failing test**

```python
import re
import numpy as np
from vectormark.pipeline import idealize, Options

def _big_logo(H=1500, W=1500):
    img = np.full((H, W, 3), 255, np.uint8)
    img[400:1100, 400:1100] = (200, 40, 40)     # big red square
    return img

def test_output_preserves_original_dimensions():
    svg = idealize(_big_logo(), options=Options(working_max_dim=768))
    m = re.search(r'<svg[^>]*\bwidth="(\d+)"\s+height="(\d+)"', svg)
    assert m and (int(m.group(1)), int(m.group(2))) == (1500, 1500)
    vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert vb and max(int(vb.group(1)), int(vb.group(2))) == 768   # working space

def test_small_input_identical_to_disabled():
    img = np.full((300, 300, 3), 255, np.uint8); img[80:220, 80:220] = (20, 40, 200)
    assert idealize(img, options=Options(working_max_dim=768)) == \
           idealize(img, options=Options(working_max_dim=None))

def test_conditioned_idealize_is_deterministic():
    big = _big_logo()
    assert idealize(big, options=Options(working_max_dim=512)) == \
           idealize(big, options=Options(working_max_dim=512))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_conditioning.py -k "dimensions or identical or deterministic" -v`
Expected: FAIL (output width=768 not 1500, no conditioning wired).

- [ ] **Step 3: Implement**

In `idealize`, after the flatten block that produces `arr` (right before `h0, w0 = arr.shape[:2]`):
```python
    orig_h, orig_w = arr.shape[:2]
    arr = _condition_input(arr, opt.working_max_dim)
```
Leave `h0, w0 = arr.shape[:2]` as-is (now conditioned dims — correct for all downstream processing and rectify).

Just before the final `return`, normalize the output size:
```python
    if (arr.shape[1], arr.shape[0]) != (orig_w, orig_h):
        svg = _set_svg_output_size(svg, orig_w, orig_h)
```

Add the helper:
```python
def _set_svg_output_size(svg: str, width: int, height: int) -> str:
    """Rewrite the <svg> element's width/height attributes to the original input size,
    leaving viewBox (working space) intact — a pure display scale (SVG is resolution-free)."""
    return re.sub(r'(<svg\b[^>]*?)\bwidth="\d+"\s+height="\d+"',
                  rf'\1width="{width}" height="{height}"', svg, count=1)
```
Ensure `import re` is present at module top (add if missing).

- [ ] **Step 4: Run tests + full suite**

Run: `uv run pytest tests/test_conditioning.py -v` → PASS.
Run: `uv run pytest -q` → green. The synthetic golden tests are all ≤500px → untouched (conditioning no-ops). If any golden shifts, STOP and report (it should not).

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/pipeline.py tests/test_conditioning.py
git commit -m "feat(pipeline): condition oversized inputs in idealize; preserve original output dimensions"
```

---

### Task 3: Acceptance — V-bird improvement + large-corpus non-regression

**Files:**
- Test: `tests/test_conditioning_acceptance.py` (create)

**Interfaces:**
- Consumes: `idealize` with conditioning (Tasks 1-2).

- [ ] **Step 1: Write the acceptance test**

```python
import os
import numpy as np
import pytest
from PIL import Image
from vectormark.pipeline import idealize, Options

VBIRD = os.path.join(os.path.dirname(__file__), "..", "scratch", "real-logos", "vbird.png")

@pytest.mark.skipif(not os.path.exists(VBIRD), reason="V-bird not present")
def test_vbird_conditioned_emits_more_circles_than_native():
    arr = np.asarray(Image.open(VBIRD).convert("RGB"), np.uint8)
    native = idealize(arr, options=Options(working_max_dim=None))
    cond = idealize(arr, options=Options(working_max_dim=512))
    # conditioning recovers at least one more round dot
    assert cond.count("<circle") + cond.count("<ellipse") >= \
           native.count("<circle") + native.count("<ellipse")
    # and does not explode path count (no fraying)
    assert cond.count("<path") <= native.count("<path") + 2

def test_default_conditioning_keeps_small_logo_valid():
    img = np.full((480, 480, 3), 255, np.uint8); img[120:360, 120:360] = (30, 120, 240)
    svg = idealize(img)   # default working_max_dim=768, 480<768 -> pass-through
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
    assert "<rect" in svg or "<polygon" in svg or "<path" in svg
```

- [ ] **Step 2: Run + full suite**

Run: `uv run pytest tests/test_conditioning_acceptance.py -v` → PASS (V-bird test skips if image absent).
Run: `uv run pytest -q` → green.

Manual smoke (report only, do not gate): idealize burger_king.png (1280×1395) and confirm it still emits its expected shapes with conditioning on (downscaled to 768) — note any shape loss in the report; if a real regression, STOP and report rather than committing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_conditioning_acceptance.py
git commit -m "test(conditioning): V-bird improvement + small-logo pass-through acceptance"
```
