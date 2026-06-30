import numpy as np
from vectormark.contour import outer_contour
from vectormark.fit import (
    MAX_PATH_SEGMENTS, MAX_POLY_VERTICES,
    fit_path, recognize_polygon, recognize_primitive,
)


def _disk(cx, cy, r, size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _rect(x0, y0, x1, y1, size=60):
    m = np.zeros((size, size), bool); m[y0:y1, x0:x1] = True
    return m


def test_recognizes_circle():
    c = outer_contour(_disk(30, 30, 18))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "circle"
    assert abs(shape.params["r"] - 18) < 1.5


def test_recognizes_axis_aligned_rect():
    c = outer_contour(_rect(10, 14, 50, 40))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "rect"
    assert abs(shape.params["w"] - 40) < 2 and abs(shape.params["h"] - 26) < 2


def test_rejects_half_ellipse_region():
    # dome = ellipse with the bottom flattened -> NOT a whole primitive
    yy, xx = np.ogrid[:60, :60]
    dome = (((xx - 30) ** 2 / 324 + (yy - 40) ** 2 / 400) <= 1) & (yy <= 40)
    c = outer_contour(dome)
    assert recognize_primitive(c, epsilon=1.0) is None


def _trapezoid(size=60):
    m = np.zeros((size, size), bool)
    for y in range(10, 40):
        half = int(20 - (y - 10) * 0.3)
        m[y, 30 - half:30 + half] = True
    return m


def test_recognizes_trapezoid_as_polygon():
    c = outer_contour(_trapezoid())
    shape = recognize_polygon(c, epsilon=1.2)
    assert shape is not None and shape.kind == "polygon"
    assert 4 <= len(shape.params["points"]) <= 5


def test_polygon_rejects_curved_region():
    c = outer_contour(_disk_local())
    assert recognize_polygon(c, epsilon=1.2) is None


def _disk_local(size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - 30) ** 2 + (yy - 30) ** 2) <= 18 * 18


def test_fit_path_of_square_uses_only_lines():
    mask = np.zeros((30, 30), bool); mask[6:24, 6:24] = True
    c = outer_contour(mask)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    assert shape is not None
    assert shape.kind == "path"
    assert "C" not in shape.params["d"]      # all straight -> only line ops
    assert shape.params["d"].strip().endswith("Z")


def test_fit_path_of_dome_uses_curve():
    yy, xx = np.ogrid[:80, :80]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 1600) <= 1) & (yy <= 55)
    c = outer_contour(dome)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    # curved top -> inflection-free quadratic arcs (Q), never cubic (C) by default
    assert shape is not None
    assert shape.kind == "path" and "Q" in shape.params["d"] and "C" not in shape.params["d"]
    # opt-in cubic flag flips emission to cubics (C), no quadratics
    cshape = fit_path(c, epsilon=1.0, max_error=0.8, cubic=True, max_segments=32)
    assert cshape is not None
    assert cshape.kind == "path" and "C" in cshape.params["d"] and "Q" not in cshape.params["d"]


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


def test_vertex_cap_rejects_too_many_corners():
    # _square() produces 3 corner-runs; a cap of 2 must reject it via the vertex gate
    # (segment count is tiny, so this exercises the vertex cap in isolation)
    assert fit_path(_square(), epsilon=1.5, max_error=1.0, max_vertices=2) is None


def test_polygon_default_bound_is_the_constant():
    # a clean hexagon (6 verts) is within the bound -> recognized
    t = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexa = np.column_stack([50 + 40 * np.cos(t), 50 + 40 * np.sin(t)])
    ring = np.vstack([hexa, hexa[0]])
    shp = recognize_polygon(ring, epsilon=1.5)
    assert shp is not None and 3 <= len(shp.params["points"]) <= MAX_POLY_VERTICES
    # forcing a 2-vertex bound rejects it
    assert recognize_polygon(ring, epsilon=1.5, max_vertices=2) is None


# ── Task-5 tests: robust (RMS) recognition acceptance ────────────────────────

def _disc_mask(r=20, noise=False, seed=0):
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    m = ((yy - 40) ** 2 + (xx - 40) ** 2) <= r ** 2
    if noise:  # erode a noisy 2px antialiased ring, like a quantized dot
        rng = np.random.default_rng(seed)
        ring = (((yy - 40) ** 2 + (xx - 40) ** 2) <= r ** 2) & ~(((yy - 40) ** 2 + (xx - 40) ** 2) <= (r - 2) ** 2)
        m = m & ~(ring & (rng.random((H, W)) < 0.5))
    return m


def test_noisy_disc_recovers_as_circle():
    # the eroded/quantized dot must recognize as a circle (fit to the bulk), not be rejected
    c = outer_contour(_disc_mask(noise=True))
    shp = recognize_primitive(c, epsilon=1.5)
    assert shp is not None and shp.kind == "circle"


def test_clean_disc_still_circle():
    shp = recognize_primitive(outer_contour(_disc_mask()), epsilon=1.5)
    assert shp is not None and shp.kind == "circle"


def test_square_is_not_accepted_as_circle():
    # robustness must NOT over-accept: a square's points are far from any circle
    H = W = 80
    m = np.zeros((H, W), bool); m[20:60, 20:60] = True
    shp = recognize_primitive(outer_contour(m), epsilon=1.5)
    assert shp is None or shp.kind in ("rect", "ellipse")  # never 'circle'
