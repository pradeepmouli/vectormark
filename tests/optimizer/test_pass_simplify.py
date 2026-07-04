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


def test_simplify_pass_does_not_worsen_cutout_subpath_when_outer_improves():
    outer = (
        "M387 497.5 Q318 497.5 249 497.5 Q208.5 497.5 168 497.5 "
        "Q138.48 498.25 110 496.5 Q82.94 495.98 60 485.5 "
        "Q49.76 479.58 40 472.5 Q3.21 440.02 1.5 393 "
        "Q-0.25 381.52 0.5 369 Q-1.25 341.02 -0.5 312 "
        "Q-0.5 266.5 -0.5 221 Q-0.5 168 -0.5 115 "
        "Q1.2 104.79 1.5 94 Q4.32 81.93 7.5 70 "
        "Q13.38 56.53 21.5 44 Q56.67 0.93 111 0.5 "
        "Q179.49 -1.24 249 -0.5 Q289.5 -0.5 330 -0.5 "
        "Q357.02 -1.25 383 0.5 Q410.99 0.69 435 10.5 "
        "Q445.1 16.25 455 22.5 Q494.59 55.21 496.5 105 "
        "Q498.25 116.98 497.5 130 Q497.5 157.5 497.5 185 "
        "Q497.5 231 497.5 277 Q498.19 337.99 496.5 398 "
        "Q493.56 414.94 488.5 431 Q483.32 441.13 477.5 451 "
        "Q447.21 491.42 399 495.5 Q393 496.5 387 497.5 Z"
    )
    cutout = (
        "M195.5 193 Q197.12 187.95 202 185.5 Q269.25 172.6 336 158.5 "
        "Q339.11 158.73 342 159.5 Q343.47 161.16 344.5 163 "
        "Q344.5 224.5 344.5 286 Q340.08 293.48 332 295.5 "
        "Q321.61 296.57 312 299.5 Q283.43 302.1 269.5 327 "
        "Q267.93 334.47 266.5 342 Q267.82 349.08 269.5 356 "
        "Q283.68 384.52 317 380.5 Q327.06 378.14 337 375.5 "
        "Q357.18 365.14 363.5 344 Q362.84 341.02 364.5 339 "
        "Q364.5 212 364.5 85 Q362.62 78.19 356 75.5 L183 110.5 "
        "Q178.7 114.31 176.5 119 Q176.5 217 176.5 315 "
        "Q174.62 318.84 173.5 323 Q167.26 329.19 159 330.5 "
        "Q151.11 331.06 144 333.5 Q120.9 335.16 104.5 353 "
        "Q99.53 361.89 97.5 372 Q97.39 401.61 124 413.5 "
        "Q132.21 415.21 141 415.5 Q145.5 414.5 150 413.5 "
        "Q170.03 411.95 184.5 397 Q191.41 386.79 194.5 375 "
        "Q196.24 284.51 195.5 193 Z"
    )
    obj = _obj_from_path(Shape("path", {"d": f"{outer} {cutout}", "fill_rule": "evenodd"}))

    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj, (512, 512))})
    simplified_d = str(proposals[0].new_objects[0].current.params["d"])

    assert _command_count(simplified_d) < _command_count(f"{outer} {cutout}")
    assert "L183 110.5" in simplified_d
    assert "L344.5 286" in simplified_d
    assert "Q240.66 46.53" not in simplified_d
    assert "Q344.5 224.5 344.5 286" not in simplified_d


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
