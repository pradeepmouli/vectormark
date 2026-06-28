# Inflection-Guarded Cubic Bézier Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Fit curved boundary runs with inflection-guarded cubic Béziers (strictly more expressive than quadratics, never inflecting), replacing quadratic-only path fitting — so smooth curves fit in far fewer segments and stop blowing the budget / cutting corners.

**Architecture:** Extend `_fitcurve.py` (already Schneider-structured) with a cubic basis + 2-control least-squares + an exact inflection guard that splits-and-recurses, so every emitted cubic is single-curvature. `fit_path` emits `C` instead of `Q`. Endpoint-first API so the seam-graph reuses it.

**Tech Stack:** numpy.

## Global Constraints

- Python ≥ 3.12, pure-Python. TDD. `rg` not `grep`. Determinism: chord-length parameterization, deterministic split (worst-deviation index), no RNG.
- **Every emitted cubic MUST be inflection-free** — the guard is mandatory; an inflecting cubic is never returned.
- Do NOT `git add scratch/`. Commit trailer EXACTLY: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.
- Pre-existing `test_stdio_server_exposes_idealize_logo_tool` (MCP) failure, if present, is not yours.

---

### Task 1: Cubic fitter + inflection guard in `_fitcurve.py`

**Files:**
- Modify: `src/vectormark/_fitcurve.py`
- Test: `tests/test_fitcurve_cubic.py` (create)

**Interfaces:**
- Produces:
  - `cubic_inflects(ctrl: np.ndarray) -> bool` — True iff the cubic `ctrl` (shape (4,2)) changes curvature sign in (0,1).
  - `fit_cubic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]` — list of (4,2) cubic control arrays `[P0,P1,P2,P3]`, each within `max_error` of the data AND inflection-free (splits/recurses to guarantee both). Endpoints pinned to `points[0]`/`points[-1]`; adjacent cubics share endpoints.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from vectormark._fitcurve import fit_cubic_beziers, cubic_inflects

def _arc(n=40, deg=90, r=200):
    th = np.linspace(0, np.deg2rad(deg), n)
    return np.c_[r*np.cos(th), r*np.sin(th)] + [60, 40]

def _scurve(n=60):
    x = np.linspace(0, 360, n); y = 90*np.sin(np.deg2rad(x))
    return np.c_[x+40, y+200]

def test_arc_fits_in_one_cubic_low_error():
    cs = fit_cubic_beziers(_arc(), max_error=2.0)
    assert len(cs) == 1
    assert not cubic_inflects(cs[0])

def test_every_emitted_cubic_is_inflection_free():
    for data in (_arc(), _scurve(), _arc(deg=170)):
        for c in fit_cubic_beziers(data, max_error=1.5):
            assert not cubic_inflects(c), "emitted a cubic with an inflection point"

def test_scurve_is_split_not_inflected():
    cs = fit_cubic_beziers(_scurve(), max_error=1.5)
    assert len(cs) >= 2     # the S is split into convex pieces

def test_endpoints_are_pinned_and_shared():
    data = _scurve(); cs = fit_cubic_beziers(data, max_error=1.5)
    assert np.allclose(cs[0][0], data[0]) and np.allclose(cs[-1][3], data[-1])
    for a, b in zip(cs, cs[1:]):
        assert np.allclose(a[3], b[0])     # contiguous

def test_deterministic():
    d = _scurve()
    a = fit_cubic_beziers(d, 1.5); b = fit_cubic_beziers(d, 1.5)
    assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fitcurve_cubic.py -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement** (append to `_fitcurve.py`; reuse existing `_chord_length_parameterize`)

```python
def cbezier(ctrl, t):
    """Cubic Bézier point(s) at parameter t."""
    mt = 1 - t
    return (mt**3 * ctrl[0] + 3*mt**2*t * ctrl[1] + 3*mt*t**2 * ctrl[2] + t**3 * ctrl[3])


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def cubic_inflects(ctrl: np.ndarray) -> bool:
    """True iff the planar cubic changes curvature sign in (0,1) — an inflection / S-curve.
    Sampled sign-change of cross(B'(t), B''(t)); robust and deterministic."""
    ctrl = np.asarray(ctrl, float)
    a, b, c = ctrl[1] - ctrl[0], ctrl[2] - ctrl[1], ctrl[3] - ctrl[2]   # first differences
    t = np.linspace(0.02, 0.98, 64)[:, None]
    d1 = 3 * ((1 - t)**2 * a + 2 * (1 - t) * t * b + t**2 * c)          # B'(t)
    d2 = 6 * ((1 - t) * (b - a) + t * (c - b))                          # B''(t)
    s = np.sign(_cross(d1, d2)); s = s[s != 0]
    return bool(np.any(np.diff(s) != 0))


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else np.array([0.0, 0.0])


def _generate_cubic(pts, u, t0, t3):
    """Schneider least-squares: endpoints pinned, end tangents fixed; solve the two tangent
    magnitudes a0,a3 (Graphics Gems 1990). P1 = P0 + a0*t0, P2 = P3 + a3*t3."""
    P0, P3 = pts[0], pts[-1]
    b0 = (1 - u)**3; b1 = 3*(1 - u)**2*u; b2 = 3*(1 - u)*u**2; b3 = u**3
    A1 = t0[None, :] * b1[:, None]; A2 = t3[None, :] * b2[:, None]
    f = pts - (P0 * (b0 + b1)[:, None] + P3 * (b2 + b3)[:, None])
    c11 = float((A1 * A1).sum()); c12 = float((A1 * A2).sum()); c22 = float((A2 * A2).sum())
    x1 = float((f * A1).sum()); x2 = float((f * A2).sum())
    det = c11 * c22 - c12 * c12
    chord = float(np.hypot(*(P3 - P0)))
    if abs(det) < 1e-12:
        a0 = a3 = chord / 3.0
    else:
        a0 = (x1 * c22 - x2 * c12) / det
        a3 = (c11 * x2 - c12 * x1) / det
    # keep control points on the tangent rays (no negative/backward magnitudes)
    lo = chord * 1e-2
    a0 = a0 if a0 > lo else chord / 3.0
    a3 = a3 if a3 > lo else chord / 3.0
    return np.array([P0, P0 + a0 * t0, P3 + a3 * t3, P3])


def _max_error_cubic(pts, ctrl, u):
    dev = np.linalg.norm(cbezier(ctrl, u[:, None]) - pts, axis=1)
    i = int(np.argmax(dev))
    return float(dev[i]), i


def fit_cubic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return []
    t0 = _unit(pts[1] - pts[0]); t3 = _unit(pts[-2] - pts[-1])
    return _fit_cubic(pts, max_error, t0, t3)


def _fit_cubic(pts, error, t0, t3):
    if len(pts) == 2:
        P0, P3 = pts[0], pts[1]
        d = (P3 - P0) / 3.0
        return [np.array([P0, P0 + d, P3 - d, P3])]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_cubic(pts, u, t0, t3)
    max_err, split = _max_error_cubic(pts, ctrl, u)
    if max_err < error and not cubic_inflects(ctrl):
        return [ctrl]
    if max_err < error * error and not cubic_inflects(ctrl):
        # Newton reparameterize a few times before giving up (error close)
        for _ in range(8):
            u = np.array([_newton_cubic(p, uu, ctrl) for p, uu in zip(pts, u)])
            ctrl = _generate_cubic(pts, u, t0, t3)
            max_err, split = _max_error_cubic(pts, ctrl, u)
            if max_err < error and not cubic_inflects(ctrl):
                return [ctrl]
    if split <= 0 or split >= len(pts) - 1:
        split = len(pts) // 2
    # interior tangent at the split (centered difference) shared by both halves
    tm = _unit(pts[split + 1] - pts[split - 1])
    left = _fit_cubic(pts[: split + 1], error, t0, -tm)
    right = _fit_cubic(pts[split:], error, tm, t3)
    return left + right


def _newton_cubic(p, u, ctrl):
    mt = 1 - u
    d = cbezier(ctrl, u) - p
    q1 = 3*mt**2*(ctrl[1]-ctrl[0]) + 6*mt*u*(ctrl[2]-ctrl[1]) + 3*u**2*(ctrl[3]-ctrl[2])
    q2 = 6*mt*(ctrl[2]-2*ctrl[1]+ctrl[0]) + 6*u*(ctrl[3]-2*ctrl[2]+ctrl[1])
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fitcurve_cubic.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/_fitcurve.py tests/test_fitcurve_cubic.py
git commit -m "feat(_fitcurve): inflection-guarded cubic Bézier fitting (Schneider + guard-split)"
```

---

### Task 2: Emit cubics from `fit_path`

**Files:**
- Modify: `src/vectormark/fit.py` (import + the curve-emit branch in `_build_corner_path`, ~line 155-159)
- Test: `tests/test_fit.py` (extend)

**Interfaces:**
- Consumes: `fit_cubic_beziers` (Task 1).
- Produces: `fit_path` emits `C` (cubic) commands for curved runs; straight runs still emit `L`. Output curves are smoother and use fewer commands.

- [ ] **Step 1: Write the failing test**

```python
import re
import numpy as np
from vectormark.fit import fit_path

def _disk_contour(r=40, cx=50, cy=50, n=200):
    th = np.linspace(0, 2*np.pi, n)
    return np.c_[cx + r*np.cos(th), cy + r*np.sin(th)]

def test_fit_path_emits_cubics_for_curves():
    s = fit_path(_disk_contour(), epsilon=1.0, max_error=1.5)
    assert s is not None and "C" in s.params["d"]      # cubic commands now emitted
    assert "Q" not in s.params["d"]                    # no quadratics

def test_fit_path_curve_is_compact():
    d = fit_path(_disk_contour(), epsilon=1.0, max_error=1.5).params["d"]
    assert d.count("C") <= 8                            # a near-circle is a handful of cubics
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fit.py -k "emits_cubics or curve_is_compact" -v`
Expected: FAIL (still emits Q).

- [ ] **Step 3: Implement**

In `src/vectormark/fit.py`:
- Change the import (line 11): `from ._fitcurve import fit_cubic_beziers`.
- In `_build_corner_path`, replace the non-straight branch (the `for b in fit_quadratic_beziers(seg, max_error):` loop emitting `Q{...} {...} {...}`) with:
```python
            for b in fit_cubic_beziers(seg, max_error):
                d += (f"C{_fmt(b[1][0])} {_fmt(b[1][1])} "
                      f"{_fmt(b[2][0])} {_fmt(b[2][1])} "
                      f"{_fmt(b[3][0])} {_fmt(b[3][1])} ")
                segs += 1
```
(Each cubic is one `C` drawing command; `segs += 1` per cubic, same budget accounting.)

- [ ] **Step 4: Run tests + full suite**

Run: `uv run pytest tests/test_fit.py -v` → PASS.
Run: `uv run pytest -q` → green EXCEPT expected golden shifts (curves now cubic). For each shifted real-logo/golden fixture: re-derive ONLY if element STRUCTURE is preserved (same shape kinds/counts via `re.findall(r'<(\w+)', svg)`) and command count did not grow; list each in the report; STOP+report if any shape is lost or a path gains commands (regression) or any non-golden test fails. The bounded-grammar per-subpath budget must still hold.

- [ ] **Step 5: Commit**

```bash
git add src/vectormark/fit.py tests/test_fit.py   # + re-derived goldens
git commit -m "feat(fit): emit inflection-guarded cubics from fit_path (was quadratic-only)"
```

---

### Task 3: Acceptance — no-inflection invariant + bottom-gap improvement

**Files:**
- Test: `tests/test_cubics_acceptance.py` (create)

**Interfaces:**
- Consumes: `idealize`, `fit_path`, `fit_cubic_beziers`/`cubic_inflects`.

- [ ] **Step 1: Write the acceptance test**

```python
import re, os
import numpy as np
import pytest
from PIL import Image
from vectormark._fitcurve import fit_cubic_beziers, cubic_inflects
from vectormark.pipeline import idealize, Options

def test_random_convex_and_concave_runs_never_inflect_after_fit():
    rng_pts = [
        np.c_[np.linspace(0, 100, 30), 40*np.sin(np.linspace(0, np.pi, 30))],      # convex bump
        np.c_[np.linspace(0, 100, 30), -30*np.sin(np.linspace(0, np.pi, 30))],     # concave
        np.c_[np.linspace(0, 120, 40), 50*np.sin(np.linspace(0, 2*np.pi, 40))],    # S (must split)
    ]
    for pts in rng_pts:
        for c in fit_cubic_beziers(pts, max_error=1.5):
            assert not cubic_inflects(c)

VBIRD = os.path.join(os.path.dirname(__file__), "..", "scratch", "real-logos", "vbird.png")

@pytest.mark.skipif(not os.path.exists(VBIRD), reason="V-bird not present")
def test_vbird_conditioned_uses_cubics_and_stays_bounded():
    arr = np.asarray(Image.open(VBIRD).convert("RGB"), np.uint8)
    svg = idealize(arr, options=Options(working_max_dim=512))
    assert "C" in svg                                  # cubics in real output
    # no single subpath is a frayed mega-trace
    worst = max((sum(sub.count(ch) for ch in "LCQ") for d in re.findall(r'd="([^"]*)"', svg)
                 for sub in re.split(r"(?=M)", d)), default=0)
    assert worst <= 12
```

- [ ] **Step 2: Run + full suite**

Run: `uv run pytest tests/test_cubics_acceptance.py -v` → PASS (V-bird test skips if absent; controller verifies V-bird manually).
Run: `uv run pytest -q` → green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cubics_acceptance.py
git commit -m "test(cubics): no-inflection invariant + V-bird cubic/bounded acceptance"
```
