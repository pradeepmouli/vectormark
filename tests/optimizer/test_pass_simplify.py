import math

import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.optimizer.passes.simplify import simplify_pass
from vectormark.pipeline import Options, _optimizer_passes


def _command_count(d: str) -> int:
    return sum(1 for ch in d if ch in "MLQCAZ")


def _obj_from_d(d: str, obj_id: int = 1) -> VectorRegion:
    return VectorRegion(obj_id, Shape("path", {"d": d}), FlatFill("#000000"), 0)


def _obj_from_path(shape: Shape, obj_id: int = 1) -> VectorRegion:
    return VectorRegion(obj_id, shape, FlatFill("#000000"), 0)


def _mask_for_obj(obj: VectorRegion, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(obj.footprint, shape_hw)


def _optimized_obj(
    obj: VectorRegion,
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> VectorRegion:
    out = optimize(
        [obj],
        {obj.id: _mask_for_obj(obj)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)],
    )
    return out[0]


def _optimized_d(
    obj: VectorRegion,
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> str:
    return str(_optimized_obj(obj, epsilon=epsilon, max_error=max_error, cubic=cubic).current.params["d"])


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
    simplified_d = str(optimized.current.params["d"])

    assert optimized.current.params.get("fill_rule") == "evenodd"
    assert simplified_d.count("M") == 2
    assert simplified_d.count("Q") == 8
    assert "L50 35" in simplified_d


def test_simplify_pass_drops_cutout_when_later_path_covers_it():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    background = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}), obj_id=1)
    cover = VectorRegion(2, Shape("path", {"d": hole}), FlatFill("#FFFFFF"), 1)

    out = optimize(
        [background, cover],
        {background.id: _mask_for_obj(background), cover.id: _mask_for_obj(cover)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=1.0, max_error=1.0)],
    )

    by_id = {obj.id: obj for obj in out}
    background_d = str(by_id[1].current.params["d"])
    assert by_id[1].current.params.get("fill_rule") is None
    assert background_d.count("M") == 1
    assert background_d.count("Q") == 8
    assert by_id[2].current.params["d"] == hole


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


def test_simplify_pass_tries_quadratics_before_cubics():
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

    quadratic_d = _optimized_d(obj, epsilon=1.0, max_error=1.0, cubic=False)
    cubic_enabled_d = _optimized_d(obj, epsilon=1.0, max_error=1.0, cubic=True)

    assert _command_count(cubic_enabled_d) < _command_count(d)
    assert _command_count(cubic_enabled_d) <= _command_count(quadratic_d)
    assert len(cubic_enabled_d.encode()) <= len(quadratic_d.encode())
    assert "Q" in cubic_enabled_d


def test_optimizer_passes_forward_cubic_path_option_to_simplify():
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
    simplify_quadratic = _optimizer_passes(Options(optimizer=True, cubic_paths=False))[-1]
    simplify_cubic = _optimizer_passes(Options(optimizer=True, cubic_paths=True))[-1]

    quadratic_proposals = simplify_quadratic([obj], {obj.id: _mask_for_obj(obj)})
    cubic_proposals = simplify_cubic([obj], {obj.id: _mask_for_obj(obj)})

    assert quadratic_proposals
    assert cubic_proposals
    quadratic_d = str(quadratic_proposals[0].new_objects[0].current.params["d"])
    cubic_enabled_d = str(cubic_proposals[0].new_objects[0].current.params["d"])
    assert len(cubic_enabled_d.encode()) <= len(quadratic_d.encode())


def test_simplify_pass_rejects_paths_more_complex_than_original_trace():
    current = Shape("path", {"d": _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)})
    original = Shape("path", {"d": "M8 8 L88 8 L88 88 L8 88 Z"})
    obj = VectorRegion(1, current, FlatFill("#000000"), 0, original=original)

    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj)}, epsilon=1.0, max_error=1.0)

    assert proposals == []


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

    assert out[0].current == obj.current
    assert Polygon(out[0].footprint).equals_exact(Polygon(obj.footprint), tolerance=0.0)
