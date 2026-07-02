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


def _obj_from_path(shape: Shape, obj_id: int = 1) -> OptObject:
    return OptObject(obj_id, shape, FlatFill("#000000"), 0)


def _mask_for_obj(obj: OptObject, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(obj.flat, shape_hw)


def _optimized_obj(obj: OptObject, *, epsilon: float = 1.5, max_error: float = 1.0) -> OptObject:
    out = optimize(
        [obj],
        {obj.id: _mask_for_obj(obj)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=epsilon, max_error=max_error)],
    )
    return out[0]


def _optimized_d(obj: OptObject, *, epsilon: float = 1.5, max_error: float = 1.0) -> str:
    return str(_optimized_obj(obj, epsilon=epsilon, max_error=max_error).exact.params["d"])


def _dense_rounded_rect_d(x0: float, y0: float, x1: float, y1: float, radius: float, *, samples: int = 10) -> str:
    commands = [f"M{x0 + radius} {y0}", f"L{x1 - radius} {y0}"]
    arcs = (
        (x1 - radius, y0 + radius, -math.pi / 2.0, 0.0),
        (x1 - radius, y1 - radius, 0.0, math.pi / 2.0),
        (x0 + radius, y1 - radius, math.pi / 2.0, math.pi),
        (x0 + radius, y0 + radius, math.pi, math.pi * 1.5),
    )
    straight_ends = (
        (x1, y1 - radius),
        (x0 + radius, y1),
        (x0, y0 + radius),
        (x0 + radius, y0),
    )
    for (cx, cy, start, end), line_end in zip(arcs, straight_ends, strict=True):
        for theta in np.linspace(start, end, samples + 1)[1:]:
            commands.append(f"L{cx + radius * math.cos(float(theta))} {cy + radius * math.sin(float(theta))}")
        if line_end != straight_ends[-1]:
            commands.append(f"L{line_end[0]} {line_end[1]}")
    commands.append("Z")
    return " ".join(commands)


def test_simplify_pass_refits_dense_rounded_rect_to_quadratic_border():
    d = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) < _command_count(d)
    assert simplified_d.count("Q") == 8
    assert "C" not in simplified_d


def test_simplify_pass_preserves_holes_when_refitting_outer_border():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    obj = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}))

    optimized = _optimized_obj(obj, epsilon=1.0, max_error=1.0)
    simplified_d = str(optimized.exact.params["d"])

    assert optimized.exact.params.get("fill_rule") == "evenodd"
    assert simplified_d.count("M") == 2
    assert simplified_d.count("Q") == 8
    assert "L50 35" in simplified_d


def test_simplify_pass_drops_cutout_when_later_path_covers_it():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    background = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}), obj_id=1)
    cover = OptObject(2, Shape("path", {"d": hole}), FlatFill("#FFFFFF"), 1)

    out = optimize(
        [background, cover],
        {background.id: _mask_for_obj(background), cover.id: _mask_for_obj(cover)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=1.0, max_error=1.0)],
    )

    by_id = {obj.id: obj for obj in out}
    background_d = str(by_id[1].exact.params["d"])
    assert by_id[1].exact.params.get("fill_rule") is None
    assert background_d.count("M") == 1
    assert background_d.count("Q") == 8
    assert by_id[2].exact.params["d"] == hole


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
