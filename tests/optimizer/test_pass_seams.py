import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.passes.seams import seams_pass
from vectormark.optimizer.vector_region import VectorRegion, to_polygon
from vectormark.pipeline import Options, _optimizer_passes


def _mask(shape: Shape, shape_hw: tuple[int, int] = (80, 120)) -> np.ndarray:
    return rasterize(to_polygon(shape), shape_hw)


def test_seams_pass_closes_adjacent_path_gap_to_shared_boundary():
    left = VectorRegion(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = VectorRegion(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.8 10 L49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = optimize([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L49.8 10 L49.8 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.8 10")
    assert by_id[1].footprint.boundary.distance(by_id[2].footprint.boundary) == 0
    assert by_id[1].diagnostics["seams"]["selected"] == "midpoint"


def test_seams_pass_adjusts_primitive_and_path_to_close_gap():
    left = VectorRegion(
        1,
        Shape("path", {"d": "M10 10 L49.7 10 L49.7 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = VectorRegion(
        2,
        Shape("rect", {"x": 50.3, "y": 10.0, "w": 39.7, "h": 60.0}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L50 10 L50 70 L10 70 Z"})
    right_trace = Shape("rect", {"x": 50.0, "y": 10.0, "w": 40.0, "h": 60.0})

    out = optimize([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L50 10 L50 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["x"] == 50.0
    assert by_id[2].current.params["w"] == 40.0
    assert by_id[1].footprint.boundary.distance(by_id[2].footprint.boundary) == 0


def test_seams_pass_true_ups_nearby_slanted_vertices_to_midpoints():
    left = VectorRegion(
        1,
        Shape("path", {"d": "M10 10 L49 20 L49 70 L10 60 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = VectorRegion(
        2,
        Shape("path", {"d": "M50 20.6 L90 30 L90 80 L50 70.6 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.5 20.3 L49.5 70.3 L10 60 Z"})
    right_trace = Shape("path", {"d": "M49.5 20.3 L90 30 L90 80 L49.5 70.3 Z"})

    out = optimize([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L49.5 20.3 L49.5 70.3" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.5 20.3")
    assert "L49.5 70.3" in by_id[2].current.params["d"]
    assert by_id[1].diagnostics["seams"]["selected"] == "vertex_midpoint"


def test_seams_pass_snaps_branch_child_against_sibling_region():
    left = VectorRegion(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    branch = VectorRegion.branch(id=10, children=[left], z=0)
    right = VectorRegion(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.8 10 L49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = optimize([branch, right], {10: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    child = by_id[10].children[0]
    assert "L49.8 10 L49.8 70" in child.current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.8 10")


def test_optimizer_runs_seams_after_symmetry_before_render_inlining():
    pass_names = [getattr(pass_fn, "__name__", pass_fn.__class__.__name__) for pass_fn in _optimizer_passes(Options(optimizer=True))]

    assert pass_names[-2:] == ["symmetry_pass", "seams_pass"]
