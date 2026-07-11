import math
from pathlib import Path

import numpy as np
from PIL import Image

from vectormark.drawing_plan import parse_plan
from vectormark import drawing_refine
from vectormark.drawing_refine import drawing_summary, refine, render_drawing, root_regions
from vectormark.drawing_state import DrawingStore
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions
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
    assert report["targets"] == ({"id": "g1", "source_regions": ("r1", "r2"), "geometry": "path", "fill": "FlatFill", "z": 0.0, "diagnostics": {}},)


def test_path_local_simplify_and_seams_run_after_the_requested_path_is_fitted():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    commands = [command.id for command in trace.regions[0].trace_path.commands if command.command not in {"M", "Z"}]
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {
            "op": "set_geometry",
            "target": "r1",
            "geometry": {
                "type": "path",
                "ops": [
                    {"op": "group", "id": "s1", "commands": commands},
                    {"op": "fit", "target": "s1", "type": "quadratic"},
                    {"op": "simplify"},
                    {"op": "seams"},
                    {"op": "close"},
                ],
            },
        },
    ]})

    refined = refine(trace, root_regions(trace), plan)

    assert refined[0].current is not None and refined[0].current.kind == "path"


def test_plan_tolerances_apply_defaults_then_an_operation_override(monkeypatch):
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    commands = [command.id for command in trace.regions[0].trace_path.commands if command.command not in {"M", "Z"}]
    observed: list[tuple[float, float]] = []
    original_fit_path = drawing_refine.fit_path

    def observe_fit_path(points, *, epsilon, max_error, cubic):
        observed.append((epsilon, max_error))
        return original_fit_path(points, epsilon=epsilon, max_error=max_error, cubic=cubic)

    monkeypatch.setattr(drawing_refine, "fit_path", observe_fit_path)
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
        "defaults": {"epsilon": 0.25, "max_error": 0.75}, "ops": [
            {"op": "set_geometry", "target": "r1", "epsilon": 0.5, "geometry": {"type": "path", "ops": [
                {"op": "group", "id": "s1", "commands": commands},
                {"op": "fit", "target": "s1", "type": "quadratic"},
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
