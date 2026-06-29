import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.optobject import OptObject, flatten_points, to_polygon


def test_to_polygon_circle_area_matches():
    circ = Shape("circle", {"cx": 50.0, "cy": 50.0, "r": 20.0})
    poly = to_polygon(circ, samples=64)
    assert abs(poly.area - np.pi * 20**2) / (np.pi * 20**2) < 0.01


def test_to_polygon_path_with_hole():
    d = "M0 0 L40 0 L40 40 L0 40 Z M10 10 L30 10 L30 30 L10 30 Z"
    poly = to_polygon(Shape("path", {"d": d}))
    assert abs(poly.area - (40 * 40 - 20 * 20)) < 4


def test_optobject_with_exact_refreshes_flat():
    o = OptObject(
        id=1,
        exact=Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}),
        fill=FlatFill("#000000"),
        z=0,
    )
    assert abs(o.flat.area - 100) < 1
    o2 = o.with_exact(Shape("rect", {"x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0}))
    assert abs(o2.flat.area - 200) < 1 and abs(o.flat.area - 100) < 1
