import numpy as np

from vectormark.drawing_plan import parse_plan
from vectormark.drawing_refine import refine, root_scene
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions


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


def test_refine_forces_circle_and_flat_fill_without_mutating_trace():
    trace = PythonTraceEngine().trace(_disk(), TraceOptions())
    original = trace.regions[0].trace_path.d
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}},
        {"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "#FF6600"}},
    ]})

    scene = refine(trace, root_scene(trace), plan)

    assert '<circle id="r1"' in scene.svg
    assert 'fill="#FF6600"' in scene.svg
    assert trace.regions[0].trace_path.d == original


def test_refine_uses_declared_z_order():
    trace = PythonTraceEngine().trace(_two_regions(), TraceOptions())
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0", "ops": [
        {"op": "set_z_order", "targets": ["r2", "r1"]},
    ]})

    scene = refine(trace, root_scene(trace), plan)

    assert scene.svg.index('id="r2"') < scene.svg.index('id="r1"')
