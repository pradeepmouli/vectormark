import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectormark.drawing_plan import parse_plan
from vectormark import drawing_refine
from vectormark.drawing_refine import drawing_summary, refine, render_drawing, root_regions
from vectormark.drawing_state import DrawingStore
from vectormark.drawing_trace import PythonTraceEngine, TraceCommand, TraceOptions, TracePath, TraceRegion, TraceResult, svg_path_commands
from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


def _disk() -> np.ndarray:
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:48, :48]
    image[(xx - 24) ** 2 + (yy - 24) ** 2 < 12 ** 2] = (0, 100, 240)
    return image


def _two_regions() -> np.ndarray:
    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    image[8:24, 4:24] = (240, 40, 20)
    image[8:24, 36:56] = (20, 100, 240)
    return image


def _gradient_bands() -> np.ndarray:
    image = np.full((48, 96, 3), 255, dtype=np.uint8)
    for x in range(16, 80):
        t = (x - 16) / 63
        image[12:36, x] = (int(20 + 60 * t), int(90 + 80 * t), int(220 + 25 * t))
    return image


def test_refine_forces_circle_and_flat_fill_without_mutating_trace():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    original = trace.regions[0].trace_path.d
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}},
        {"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "#FF6600"}},
    ]})

    regions = refine(trace, root_regions(trace), plan)
    rendered = render_drawing(trace, regions)

    assert '<circle id="r1"' in rendered.svg
    assert 'fill="#FF6600"' in rendered.svg
    assert trace.regions[0].trace_path.d == original


def test_root_regions_surface_merge_soft_gradient_bands_and_keep_raw_provenance():
    image = _gradient_bands()
    trace = PythonTraceEngine().trace(image, TraceOptions(max_colors=8))

    regions = root_regions(trace, image)
    summary = drawing_summary(trace, regions)

    assert len(regions) < len(trace.regions)
    assert any(len(region.source_regions) > 1 for region in regions)
    assert len(summary["regions"]) == len(regions)
    assert all("geometry" in region and "source_regions" in region for region in summary["regions"])


def test_drawing_summary_emits_canonical_segment_children_for_each_path_command():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    summary = drawing_summary(trace, root_regions(trace))
    geometry = summary["regions"][0]["geometry"]

    commands = [command for command in geometry["commands"] if command["command"] not in {"M", "Z"}]
    assert geometry["segments"] == [
        {"id": f"{command['id']}-1", "source_commands": [command["id"]]}
        for command in commands
    ]


def test_refine_uses_declared_z_order():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_z_order", "targets": ["r2", "r1"]},
    ]})

    regions = refine(trace, root_regions(trace), plan)
    rendered = render_drawing(trace, regions)

    assert rendered.svg.index('id="r2"') < rendered.svg.index('id="r1"')


def test_merge_creates_one_native_target_with_combined_source_provenance():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "merge", "id": "g1", "regions": ["r1", "r2"]},
        {"op": "set_z_order", "targets": ["g1"]},
    ]})

    regions = refine(trace, root_regions(trace), plan)
    report = render_drawing(trace, regions).report

    assert len(regions) == 1
    assert report["targets"] == ({"id": "g1", "source_regions": ("r1", "r2"), "geometry": "path", "fill": "FlatFill", "z": 0.0, "diagnostics": {"merge": {"unioned": True, "sources": ["r1", "r2"]}}},)


def test_merge_unions_adjacent_same_fill_geometry_before_svg_emission():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    raster = np.ones((20, 20), dtype=bool)
    left = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}), fill=FlatFill("#0064F0"),
        z=0, raster=raster, drawing_id="r1", source_regions=("r1",),
    )
    right = VectorRegion.from_shape(
        id=2, shape=Shape("path", {"d": "M10 0 L20 0 L20 10 L10 10 Z"}), fill=FlatFill("#0064F0"),
        z=1, raster=raster, drawing_id="r2", source_regions=("r2",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "merge", "id": "g1", "regions": ["r1", "r2"]},
    ]})

    merged = refine(trace, (left, right), plan)

    assert len(merged) == 1
    assert merged[0].footprint.area == 200.0
    assert len(merged[0].footprint.geoms) == 1
    assert render_drawing(trace, merged).svg.count('id="g1"') == 1


def test_merge_accepts_a_child_region_target_and_removes_it_from_its_parent_tree():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    raster = np.ones((20, 20), dtype=bool)
    child = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}), fill=FlatFill("#0064F0"),
        z=0, raster=raster, drawing_id="r1", source_regions=("r1",),
    )
    branch = VectorRegion.branch(id=10, children=[child], drawing_id="r1", source_regions=("r1",))
    other = VectorRegion.from_shape(
        id=2, shape=Shape("path", {"d": "M10 0 L20 0 L20 10 L10 10 Z"}), fill=FlatFill("#0064F0"),
        z=1, raster=raster, drawing_id="r2", source_regions=("r2",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "merge", "id": "g1", "regions": ["r1-1", "r2"]},
    ]})

    merged = refine(trace, (branch, other), plan)

    assert [region.drawing_id for region in merged] == ["g1"]
    assert merged[0].source_regions == ("r1", "r2")
    assert merged[0].footprint.area == 200.0


def test_path_local_simplify_and_stitch_run_after_the_requested_path_is_fitted():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    command = next(command for command in trace.regions[0].trace_path.commands if command.command not in {"M", "Z"})
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {
            "op": "set_geometry",
            "target": "r1",
            "geometry": {
                "type": "path",
                "ops": [
                    {"op": "fit", "target": f"{command.id}-1", "type": "quadratic"},
                    {"op": "simplify"},
                    {"op": "stitch"},
                    {"op": "close"},
                ],
            },
        },
    ]})

    refined = refine(trace, root_regions(trace), plan)

    assert refined[0].current is not None and refined[0].current.kind == "path"


def test_path_geometry_preserves_compound_subpaths_and_evenodd_fill_rule():
    path = TracePath(
        d="M0 0 L10 0 L10 10 L0 10 Z M2 2 L8 2 L8 8 L2 8 Z",
        fill_rule="evenodd",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "L", (10.0, 0.0)),
            TraceCommand("r1.p0.c2", "L", (10.0, 10.0)),
            TraceCommand("r1.p0.c3", "L", (0.0, 10.0)),
            TraceCommand("r1.p0.c4", "Z", ()),
            TraceCommand("r1.p1.c0", "M", (2.0, 2.0)),
            TraceCommand("r1.p1.c1", "L", (8.0, 2.0)),
            TraceCommand("r1.p1.c2", "L", (8.0, 8.0)),
            TraceCommand("r1.p1.c3", "L", (2.0, 8.0)),
            TraceCommand("r1.p1.c4", "Z", ()),
        ),
    )
    trace = TraceResult(
        10, 10, TraceOptions(),
        (TraceRegion("r1", 1, "#0064F0", np.ones((10, 10), dtype=bool), (), path, "pixel"),),
        "<svg/>",
    )
    root = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": path.d, "fill_rule": "evenodd"}), fill=FlatFill("#0064F0"),
        z=0, raster=np.ones((10, 10), dtype=bool), drawing_id="r1", source_regions=("r1",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [
            {"op": "fit", "target": "r1.p0.c1-1", "type": "keep"},
            {"op": "fit", "target": "r1.p0.c2-1", "type": "keep"},
            {"op": "fit", "target": "r1.p0.c3-1", "type": "keep"},
            {"op": "close"},
            {"op": "fit", "target": "r1.p1.c1-1", "type": "keep"},
            {"op": "fit", "target": "r1.p1.c2-1", "type": "keep"},
            {"op": "fit", "target": "r1.p1.c3-1", "type": "keep"},
            {"op": "close"},
        ]}},
    ]})

    refined = refine(trace, (root,), plan)

    assert refined[0].current is not None
    assert refined[0].current.params["fill_rule"] == "evenodd"
    assert refined[0].current.params["d"] == "M0 0 L10 0 L10 10 L0 10 Z M2 2 L8 2 L8 8 L2 8 Z"


def test_path_fit_addresses_a_segment_child_and_preserves_unedited_retained_geometry():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    root = root_regions(trace)[0]
    assert root.current is not None
    original = root.current.params["d"]
    line_command = next(command for command in svg_path_commands(original, "r1") if command.command == "Q")
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [
            {"op": "fit", "target": f"{line_command.id}-1", "type": "line"},
        ]}},
    ]})

    refined = refine(trace, (root,), plan)

    assert refined[0].current is not None
    assert refined[0].current.params["d"].count("M") == original.count("M")
    assert refined[0].current.params["d"].count("Z") == original.count("Z")
    assert "L" in refined[0].current.params["d"]


def test_path_match_length_uses_another_retained_segment_as_its_reference():
    path = TracePath(
        d="M0 0 L10 0 L10 4 Z",
        fill_rule="nonzero",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "L", (10.0, 0.0)),
            TraceCommand("r1.p0.c2", "L", (10.0, 4.0)),
            TraceCommand("r1.p0.c3", "Z", ()),
        ),
    )
    trace = TraceResult(
        12, 12, TraceOptions(),
        (TraceRegion("r1", 1, "#0064F0", np.ones((12, 12), dtype=bool), (), path, "pixel"),),
        "<svg/>",
    )
    root = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": path.d}), fill=FlatFill("#0064F0"), z=0,
        raster=np.ones((12, 12), dtype=bool), drawing_id="r1", source_regions=("r1",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [
            {"op": "match_length", "target": "r1.p0.c2-1", "reference": "r1.p0.c1-1"},
        ]}},
    ]})

    refined = refine(trace, (root,), plan)

    assert refined[0].current is not None
    assert refined[0].current.params["d"] == "M0 0 L10 0 L10 10 Z"


@pytest.mark.parametrize(
    ("path_op", "expected_d"),
    [
        ({"op": "match", "target": "r1.p0.c2-1", "reference": "r1.p0.c1-1", "transform": [1, 0, 0, 1, 6, 0]}, "M0 0 L6 0 L12 0 Z"),
        ({"op": "set_parallel", "target": "r1.p0.c2-1", "reference": "r1.p0.c1-1"}, "M0 0 L4.5 1.5 L7.5 1.5 Z"),
        ({"op": "align", "target": "r1.p0.c2-1", "reference": "r1.p0.c1-1", "axes": ["y"]}, "M0 0 L6 0 L6 0 Z"),
    ],
)
def test_path_constraints_reconstruct_a_segment_from_a_retained_reference(path_op, expected_d):
    path = TracePath(
        d="M0 0 L6 0 L6 3 Z",
        fill_rule="nonzero",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "L", (6.0, 0.0)),
            TraceCommand("r1.p0.c2", "L", (6.0, 3.0)),
            TraceCommand("r1.p0.c3", "Z", ()),
        ),
    )
    trace = TraceResult(
        12, 12, TraceOptions(),
        (TraceRegion("r1", 1, "#0064F0", np.ones((12, 12), dtype=bool), (), path, "pixel"),),
        "<svg/>",
    )
    root = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": path.d}), fill=FlatFill("#0064F0"), z=0,
        raster=np.ones((12, 12), dtype=bool), drawing_id="r1", source_regions=("r1",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [path_op]}},
    ]})

    refined = refine(trace, (root,), plan)

    assert refined[0].current is not None
    assert refined[0].current.params["d"] == expected_d


def test_path_remove_joins_the_surviving_segments():
    path = TracePath(
        d="M0 0 L6 0 L6 3 L0 3 Z",
        fill_rule="nonzero",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "L", (6.0, 0.0)),
            TraceCommand("r1.p0.c2", "L", (6.0, 3.0)),
            TraceCommand("r1.p0.c3", "L", (0.0, 3.0)),
            TraceCommand("r1.p0.c4", "Z", ()),
        ),
    )
    trace = TraceResult(
        12, 12, TraceOptions(),
        (TraceRegion("r1", 1, "#0064F0", np.ones((12, 12), dtype=bool), (), path, "pixel"),),
        "<svg/>",
    )
    root = VectorRegion.from_shape(
        id=1, shape=Shape("path", {"d": path.d}), fill=FlatFill("#0064F0"), z=0,
        raster=np.ones((12, 12), dtype=bool), drawing_id="r1", source_regions=("r1",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [
            {"op": "remove", "target": "r1.p0.c2-1"},
        ]}},
    ]})

    refined = refine(trace, (root,), plan)

    assert refined[0].current is not None
    assert refined[0].current.params["d"] == "M0 0 L6 0 L0 3 Z"


def test_path_remove_schedules_post_fit_simplification_unless_marked_as_a_break(monkeypatch):
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    root = root_regions(trace)[0]
    assert root.current is not None
    removable = next(command for command in svg_path_commands(root.current.params["d"], "r1") if command.command not in {"M", "Z"})
    calls: list[str] = []

    def record_smoothing(regions, trace, operation, target_id, *, epsilon, max_error):
        calls.append(operation)
        return tuple(regions)

    monkeypatch.setattr(drawing_refine, "_run_detection", record_smoothing)
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "path", "ops": [
            {"op": "remove", "target": f"{removable.id}-1"},
        ]}},
    ]})

    refine(trace, (root,), plan)

    assert calls == ["simplify"]


def test_align_centers_one_retained_region_on_another_along_selected_axes():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    left = VectorRegion.from_shape(
        id=1, shape=Shape("rect", {"x": 0, "y": 0, "w": 10, "h": 10}), fill=FlatFill("#0064F0"), z=0,
        raster=np.ones((10, 10), dtype=bool), drawing_id="r1", source_regions=("r1",),
    )
    right = VectorRegion.from_shape(
        id=2, shape=Shape("rect", {"x": 20, "y": 30, "w": 4, "h": 6}), fill=FlatFill("#F06400"), z=1,
        raster=np.ones((10, 10), dtype=bool), drawing_id="r2", source_regions=("r2",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "align", "target": "r2", "reference": "r1", "axes": ["x", "y"]},
    ]})

    refined = refine(trace, (left, right), plan)

    aligned = next(region for region in refined if region.drawing_id == "r2")
    assert aligned.current is not None
    assert aligned.current.params["d"] == "M 3 2 L 7 2 L 7 8 L 3 8 Z"


def test_plan_tolerances_apply_defaults_then_an_operation_override(monkeypatch):
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    command = next(command for command in trace.regions[0].trace_path.commands if command.command not in {"M", "Z"})
    observed: list[tuple[float, float]] = []
    original_curved_run = drawing_refine._curved_run_d

    def observe_curved_run(points, max_error, *, cubic, line_epsilon):
        observed.append((line_epsilon, max_error))
        return original_curved_run(points, max_error, cubic=cubic, line_epsilon=line_epsilon)

    monkeypatch.setattr(drawing_refine, "_curved_run_d", observe_curved_run)
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
        "defaults": {"epsilon": 0.25, "max_error": 0.75}, "ops": [
            {"op": "set_geometry", "target": "r1", "epsilon": 0.5, "geometry": {"type": "path", "ops": [
                {"op": "fit", "target": f"{command.id}-1", "type": "quadratic"},
                {"op": "close"},
            ]}},
        ]})

    refine(trace, root_regions(trace), plan)

    assert observed == [(0.5, 0.75)]


def test_daikonic_root_symmetry_can_be_detected_then_explicitly_set():
    image = np.asarray(Image.open(Path("tests/fixtures/daikonic/source.png")).convert("RGB"))
    trace = PythonTraceEngine().trace(image, TraceOptions())
    roots = root_regions(trace, image)

    detected = refine(trace, roots, parse_plan({
        "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
        "ops": [{"op": "detect_symmetry", "target": "r1"}],
    }))
    symmetric_root = next(region for region in detected if region.drawing_id == "r1")
    assert symmetric_root.is_branch and len(symmetric_root.children) == 2
    axis = symmetric_root.diagnostics["symmetry"]["axis"]
    source, target = (f"r1-{child.id}" for child in symmetric_root.children)

    refined = refine(trace, detected, parse_plan({
        "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0.0",
        "ops": [{"op": "set_symmetry", "source": source, "target": target, "axis": axis}],
    }))
    updated_root = next(region for region in refined if region.drawing_id == "r1")

    assert updated_root.children[1].diagnostics["symmetry"]["mode"] == "pair"
    reported_target = next(target for target in render_drawing(trace, detected).report["targets"] if target["id"] == source)
    assert reported_target["diagnostics"]["symmetry"]["axis"] == axis


def test_daikonic_supports_geometry_choices_then_a_targeted_symmetry_block():
    image = np.asarray(Image.open(Path("tests/fixtures/daikonic/source.png")).convert("RGB"))
    trace = PythonTraceEngine().trace(image, TraceOptions())
    roots = root_regions(trace, image)
    plan = parse_plan({
        "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
        "ops": [
            {"op": "set_geometry", "target": "r1", "geometry": {"type": "cap"}},
            {"op": "set_geometry", "target": "r2", "geometry": {"type": "rounded_rect"}},
            {"op": "set_geometry", "target": "r3", "geometry": {"type": "rounded_trapezoid"}},
            {"op": "detect_clones", "target": "r6"},
            {"op": "detect_symmetry", "target": "r1"},
            {"op": "detect_symmetry", "target": "r2"},
            {"op": "detect_symmetry", "target": "r3"},
            {"op": "detect_symmetry", "target": "r4"},
        ],
    })

    refined = refine(trace, roots, plan)
    rendered = render_drawing(trace, refined)

    assert '<rect id="r2"' in rendered.svg and 'rx="' in rendered.svg
    assert {target["id"] for target in rendered.report["targets"]} >= {"r1", "r3", "r4-4", "r7"}
    base_band = next(region for region in roots if region.drawing_id == "r2")
    refined_band = next(region for region in refined if region.drawing_id == "r2")
    assert base_band is not refined_band
    assert base_band.current is not None and base_band.current.kind == "path"
    assert refined_band.current is not None and refined_band.current.kind == "rect"
    for target_id in ("r1", "r2", "r3"):
        target = next(target for target in rendered.report["targets"] if target["id"] == target_id)
        assert target["diagnostics"]["symmetry"]["mode"] == "intrinsic"
    leaf = next(target for target in rendered.report["targets"] if target["id"] == "r7")
    assert leaf["geometry"] == "use"
    assert leaf["diagnostics"]["clones"]["matched_source"] == 6
    a, b, c, d, *_translation = leaf["diagnostics"]["clones"]["transform"]
    assert a * d - b * c < 0.0


def test_refined_regions_are_the_retained_version_state_and_svg_is_a_projection():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    regions = root_regions(trace)
    store = DrawingStore()
    session = object()
    drawing = store.create(session, trace, regions=regions)
    version = store.append(session, drawing.id, "v0", plan={}, regions=regions)

    assert all(isinstance(region, VectorRegion) for region in version.regions)
    assert render_drawing(trace, version.regions).svg == render_drawing(trace, regions).svg


def test_refine_applies_an_explicit_clone_to_native_vector_regions():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "clone", "source": "r1", "target": "r2", "transform": [1, 0, 0, 1, 32, 0]},
    ]})

    regions = refine(trace, root_regions(trace), plan)
    rendered = render_drawing(trace, regions)

    assert '<use id="r2" href="#r1"' in rendered.svg
    assert regions[1].current is not None and regions[1].current.kind == "use"


def test_detect_clones_and_set_symmetry_apply_the_existing_region_relationships():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    clones = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "detect_clones"},
    ]})

    cloned = refine(trace, root_regions(trace), clones)
    assert cloned[1].diagnostics["clones"]["accepted"] is True

    symmetry = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_symmetry", "source": "r1", "target": "r2", "axis": {"theta": math.pi / 2, "cx": 30, "cy": 16}},
    ]})
    mirrored = refine(trace, root_regions(trace), symmetry)

    assert mirrored[1].diagnostics["symmetry"]["mode"] == "pair"


def test_branch_children_are_addressable_with_the_existing_hyphenated_convention():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    root = VectorRegion.branch(
        id=1,
        children=(
            VectorRegion.from_shape(id=1, shape=Shape("circle", {"cx": 18, "cy": 24, "r": 6}),
                fill=FlatFill("#0064f0"), z=0, raster=trace.regions[0].mask),
            VectorRegion.from_shape(id=3, shape=Shape("circle", {"cx": 30, "cy": 24, "r": 6}),
                fill=FlatFill("#0064f0"), z=1, raster=trace.regions[0].mask),
        ),
        drawing_id="r1",
        source_regions=("r1",),
    )
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_fill", "target": "r1-3", "fill": {"type": "flat", "color": "#FF6600"}},
    ]})

    regions = refine(trace, (root,), plan)
    rendered = render_drawing(trace, regions)

    assert 'id="r1-1"' in rendered.svg
    assert 'id="r1-3" fill="#FF6600"' in rendered.svg


def test_detect_symmetry_exposes_refinable_hyphenated_child_regions():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[16:48, 16:48] = (20, 100, 240)
    trace = PythonTraceEngine().trace(image, TraceOptions())
    detect = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "detect_symmetry", "target": "r1"},
    ]})
    regions = refine(trace, root_regions(trace), detect)

    assert {target["id"] for target in render_drawing(trace, regions).report["targets"]} == {"r1-1", "r1-2"}

    polished = drawing_refine._run_detection(
        regions,
        trace,
        "stitch",
        "r1-1",
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
    )
    polished_root = polished[0]
    assert polished_root.children[1].current is not None
    assert polished_root.children[1].current.kind == "path"
    assert polished_root.children[1].diagnostics["symmetry"]["baked_for_polish"] is True

    fill_child = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0.0", "ops": [
        {"op": "set_fill", "target": "r1-2", "fill": {"type": "flat", "color": "#FF6600"}},
    ]})
    refined = refine(trace, regions, fill_child)

    assert 'id="r1-2"' in render_drawing(trace, refined).svg

    store = DrawingStore()
    session = object()
    drawing = store.create(session, trace, regions=root_regions(trace))
    version = store.append(session, drawing.id, "v0", plan={}, regions=refined)
    _, retained = store.get(session, drawing.id, version.id)

    assert retained.regions is not None
    assert 'id="r1-2"' in render_drawing(trace, retained.regions).svg
