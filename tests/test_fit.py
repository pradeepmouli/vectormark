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
    c = outer_contour(_disk := _disk_local())
    assert recognize_polygon(c, epsilon=1.2) is None


def _disk_local(size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - 30) ** 2 + (yy - 30) ** 2) <= 18 * 18
