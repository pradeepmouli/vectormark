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


import numpy as np
from vectormark.types import Region
from vectormark.score import render_delta_e


def _disk_region(cx, cy, r, h, w, color_hex):
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return Region(1, mask, color_hex)


def test_render_delta_e_near_zero_on_matching_flat():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                     FlatFill("#1e64eb"), "region")
    assert render_delta_e(cand, src, region) < 0.03


def test_render_delta_e_large_on_wrong_color():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    wrong = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                      FlatFill("#ff0000"), "region")
    assert render_delta_e(wrong, src, region) > 0.2


from vectormark.score import structural_priors


def _square_region(h, w, color_hex):
    mask = np.zeros((h, w), bool)
    mask[10:50, 10:50] = True
    return Region(1, mask, color_hex)


def test_priors_reject_degenerate_gradient_stop_span():
    region = _square_region(60, 60, "#202020")
    grad = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     LinearGradientFill({"x1": 10, "y1": 10, "x2": 50, "y2": 10},
                                        [(0.0, "#202020"), (1.0, "#212121")]), "gradient")
    ok, reason = structural_priors(grad, region)
    assert ok is False and "stop-span" in reason


def test_priors_pass_flat_and_primitive():
    region = _square_region(60, 60, "#202020")
    flat = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     FlatFill("#202020"), "region")
    ok, reason = structural_priors(flat, region)
    assert ok is True and reason is None
