from vectormark.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectormark.fit import Shape
from vectormark.score import parsimony_cost


def _flat(shape):
    return Candidate(shape, FlatFill("#123456"), "region")


def test_parsimony_primitive_cheaper_than_path():
    circle = _flat(Shape("circle", {"cx": 5, "cy": 5, "r": 4}))
    path = _flat(Shape("path", {"d": "M0 0 C1 1 2 2 3 3 C4 4 5 5 6 6 C7 7 8 8 9 9 Z"}))
    assert parsimony_cost(circle) < parsimony_cost(path)


def test_parsimony_flat_cheaper_than_gradient_same_geometry():
    geom = Shape("rect", {"x": 0, "y": 0, "w": 10, "h": 10})
    flat = Candidate(geom, FlatFill("#000000"), "region")
    grad = Candidate(geom, LinearGradientFill({"x1": 0, "y1": 0, "x2": 10, "y2": 0},
                                              [(0.0, "#000"), (1.0, "#fff")]), "gradient")
    assert parsimony_cost(flat) < parsimony_cost(grad)


def test_parsimony_polygon_scales_with_vertices():
    tri = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (0, 1)]}))
    hexa = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (2, 1), (1, 2), (0, 2), (-1, 1)]}))
    assert parsimony_cost(hexa) > parsimony_cost(tri)
