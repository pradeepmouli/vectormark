import math

import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.optobject import OptObject
from vectormark.optimizer.passes.simplify import simplify_pass


def _command_count(d: str) -> int:
    return sum(1 for ch in d if ch in "MLQCAZ")


def _obj_from_d(d: str, obj_id: int = 1) -> OptObject:
    return OptObject(obj_id, Shape("path", {"d": d}), FlatFill("#000000"), 0)


def _mask_for_obj(obj: OptObject, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(obj.flat, shape_hw)


def _optimized_d(obj: OptObject, *, epsilon: float = 1.5, max_error: float = 1.0) -> str:
    out = optimize(
        [obj],
        {obj.id: _mask_for_obj(obj)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=epsilon, max_error=max_error)],
    )
    return str(out[0].exact.params["d"])


def test_simplify_pass_collapses_long_straight_line_runs():
    top = " ".join(f"L{x} 10" for x in range(12, 71, 2))
    d = f"M10 10 {top} L70 40 L10 40 Z"
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj)

    assert _command_count(simplified_d) < _command_count(d)
    assert "L70 10" in simplified_d
    assert "L70 40" in simplified_d
    assert "L10 40" in simplified_d


def test_simplify_pass_reduces_sampled_smooth_arc_to_curves():
    arc_points = []
    cx, cy, radius = 40.0, 40.0, 25.0
    for theta in np.linspace(math.pi, 0.0, 33):
        x = cx + radius * math.cos(float(theta))
        y = cy - radius * math.sin(float(theta))
        arc_points.append((x, y))

    commands = [f"M{arc_points[0][0]} {arc_points[0][1]}"]
    commands.extend(f"L{x} {y}" for x, y in arc_points[1:])
    commands.extend(["L65 40", "L15 40", "Z"])
    d = " ".join(commands)
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) < _command_count(d)
    assert "Q" in simplified_d


def test_optimize_gate_rejects_simplification_that_drops_real_bump():
    d = "M10 10 L50 10 L50 30 L62 30 L62 42 L50 42 L50 60 L10 60 Z"
    obj = _obj_from_d(d)
    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj)}, epsilon=20.0)

    assert proposals

    out = optimize(
        [obj],
        {obj.id: _mask_for_obj(obj)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=20.0)],
    )

    assert out[0].exact == obj.exact
    assert Polygon(out[0].flat).equals_exact(Polygon(obj.flat), tolerance=0.0)
