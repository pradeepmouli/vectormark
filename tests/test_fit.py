import numpy as np
from vectormark.contour import outer_contour
from vectormark.fit import recognize_primitive


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


from vectormark.fit import recognize_polygon


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


from vectormark.fit import fit_path


def test_fit_path_of_square_uses_only_lines():
    mask = np.zeros((30, 30), bool); mask[6:24, 6:24] = True
    c = outer_contour(mask)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    assert shape.kind == "path"
    assert "C" not in shape.params["d"]      # all straight -> only line ops
    assert shape.params["d"].strip().endswith("Z")


def test_fit_path_of_dome_uses_curve():
    yy, xx = np.ogrid[:80, :80]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 1600) <= 1) & (yy <= 55)
    c = outer_contour(dome)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    # curved top -> inflection-free quadratic arcs (Q), never cubic (C)
    assert shape.kind == "path" and "Q" in shape.params["d"] and "C" not in shape.params["d"]
