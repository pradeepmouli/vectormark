import base64

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_trace import (
    TraceCommand,
    TraceOptions,
    TracePath,
    TraceRegion,
    TraceResult,
)
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


def _trace() -> TraceResult:
    path = TracePath(
        d="M 0 0 Q 1 0 2 0 Q 3 0 4 0 Q 5 0 6 0 Z",
        fill_rule="nonzero",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "Q", (1.0, 0.0, 2.0, 0.0)),
            TraceCommand("r1.p0.c2", "Q", (3.0, 0.0, 4.0, 0.0)),
            TraceCommand("r1.p0.c3", "Q", (5.0, 0.0, 6.0, 0.0)),
            TraceCommand("r1.p0.c4", "Z", ()),
        ),
    )
    region = TraceRegion(
        id="r1",
        source_label=1,
        color="#112233",
        mask=np.ones((1, 1), dtype=bool),
        contours=(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),),
        trace_path=path,
        effective_trace_level="pixel",
    )
    return TraceResult(1, 1, TraceOptions(), (region,), "<svg/>")


def _two_region_trace() -> TraceResult:
    r1 = _trace().regions[0]
    r2 = TraceRegion(
        id="r2",
        source_label=2,
        color="#445566",
        mask=np.ones((1, 1), dtype=bool),
        contours=(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),),
        trace_path=TracePath(
            d=r1.trace_path.d,
            fill_rule=r1.trace_path.fill_rule,
            commands=tuple(
                TraceCommand(command.id.replace("r1.", "r2."), command.command, command.values)
                for command in r1.trace_path.commands
            ),
        ),
        effective_trace_level="pixel",
    )
    return TraceResult(1, 1, TraceOptions(), (r1, r2), "<svg/>")


def _scene(*target_ids: str) -> tuple[VectorRegion, ...]:
    return tuple(
        VectorRegion.from_shape(
            id=index + 1,
            shape=Shape("path", {"d": "M0 0Z"}),
            fill=FlatFill("#112233"),
            z=index,
            raster=np.ones((1, 1), dtype=bool),
            drawing_id=target_id,
            source_regions=(target_id,),
        )
        for index, target_id in enumerate(target_ids)
    )


def _grouped_scene(*source_regions: str) -> tuple[VectorRegion, ...]:
    return (
        VectorRegion.from_shape(
            id=1,
            shape=Shape("path", {"d": "M0 0Z"}),
            fill=FlatFill("#112233"),
            z=0,
            raster=np.ones((1, 1), dtype=bool),
            drawing_id="g1",
            source_regions=source_regions,
        ),
    )


def _path_plan(commands: list[str]) -> dict[str, object]:
    return {
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_x",
        "base_version": "v0",
        "ops": [
            {
                "op": "set_geometry",
                "target": "r1",
                "geometry": {
                    "type": "path",
                    "ops": [
                        {"op": "group", "id": "s1", "commands": commands},
                        {"op": "fit", "target": "s1", "type": "quadratic"},
                        {"op": "break", "target": "s1"},
                        {"op": "close"},
                    ],
                },
            }
        ],
    }


def test_plan_accepts_path_group_fit_break_and_close():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(_path_plan(["r1.p0.c1", "r1.p0.c2"]))

    validate_plan(plan, _trace(), _scene("r1"))


def test_plan_accepts_native_detection_and_explicit_relationship_operations():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "detect_primitives", "target": "r1"},
                {"op": "detect_clones", "target": "r2"},
                {"op": "clone", "source": "r1", "target": "r2", "transform": [1, 0, 0, 1, 5, 0]},
                {"op": "set_symmetry", "source": "r1", "target": "r2", "axis": {"theta": 0, "cx": 1, "cy": 1}},
                {"op": "detect_symmetry"},
            ],
        }
    )

    validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


@pytest.mark.parametrize("operation", ["split", "detect_symmetry"])
def test_structural_operations_require_a_version_boundary(operation: str):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [{"op": operation, "target": "r1"}, {"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "#112233"}}],
        }
    )

    with pytest.raises(PlanValidationError, match="final operation"):
        validate_plan(plan, _trace(), _scene("r1"))


def test_plan_reports_pointer_for_non_contiguous_path_commands():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(_path_plan(["r1.p0.c1", "r1.p0.c3"]))

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/0/commands/1"):
        validate_plan(plan, _trace(), _scene("r1"))


def test_path_geometry_rejects_commands_owned_by_another_region():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(_path_plan(["r2.p0.c1"]))

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/0/commands/0"):
        validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_path_geometry_group_inherits_its_regions_command_provenance():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "merge", "id": "g1", "regions": ["r1", "r2"]},
                {
                    "op": "set_geometry",
                    "target": "g1",
                    "geometry": {
                        "type": "path",
                        "ops": [
                            {"op": "group", "id": "s1", "commands": ["r1.p0.c1"]},
                            {"op": "fit", "target": "s1", "type": "line"},
                            {"op": "close"},
                            {"op": "group", "id": "s2", "commands": ["r2.p0.c1"]},
                            {"op": "fit", "target": "s2", "type": "line"},
                            {"op": "close"},
                        ],
                    },
                },
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_path_geometry_base_group_inherits_its_source_regions_command_provenance():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {
                    "op": "set_geometry",
                    "target": "g1",
                    "geometry": {
                        "type": "path",
                        "ops": [
                            {"op": "group", "id": "s1", "commands": ["r1.p0.c1"]},
                            {"op": "fit", "target": "s1", "type": "line"},
                            {"op": "close"},
                            {"op": "group", "id": "s2", "commands": ["r2.p0.c1"]},
                            {"op": "fit", "target": "s2", "type": "line"},
                            {"op": "close"},
                        ],
                    },
                },
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    validate_plan(plan, _two_region_trace(), _grouped_scene("r1", "r2"))


def test_path_geometry_rejects_trace_command_reused_across_path_groups():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    payload["ops"][0]["geometry"]["ops"] = [
        {"op": "group", "id": "s1", "commands": ["r1.p0.c1"]},
        {"op": "fit", "target": "s1", "type": "line"},
        {"op": "close"},
        {"op": "group", "id": "s2", "commands": ["r1.p0.c1"]},
        {"op": "fit", "target": "s2", "type": "line"},
        {"op": "close"},
    ]

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/3/commands/0"):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))


def test_path_geometry_rejects_trace_command_reused_across_set_geometry_operations():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    payload["ops"].append(
        {
            "op": "set_geometry",
            "target": "r1",
            "geometry": {
                "type": "path",
                "ops": [
                    {"op": "group", "id": "s2", "commands": ["r1.p0.c1"]},
                    {"op": "fit", "target": "s2", "type": "line"},
                    {"op": "close"},
                ],
            },
        }
    )

    with pytest.raises(PlanValidationError, match="/ops/1/geometry/ops/0/commands/0"):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))


@pytest.mark.parametrize(
    ("payload", "pointer"),
    [
        (
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [{"op": "set_geometry", "target": "missing", "geometry": {"type": "circle"}}],
            },
            "/ops/0/target",
        ),
        (
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [
                    {"op": "merge", "id": "g1", "regions": ["r1"]},
                    {"op": "merge", "id": "g1", "regions": ["r1"]},
                ],
            },
            "/ops/1/id",
        ),
    ],
)
def test_plan_reports_operation_pointer_for_unknown_targets_and_duplicate_ids(payload, pointer):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    with pytest.raises(PlanValidationError, match=pointer):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))


def test_plan_rejects_invalid_fill_and_non_finite_numbers():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    invalid_fill = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [{"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "blue"}}],
        }
    )
    with pytest.raises(PlanValidationError, match="/ops/0/fill/color"):
        validate_plan(invalid_fill, _trace(), _scene("r1"))

    non_finite = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {
                    "op": "set_fill",
                    "target": "r1",
                    "fill": {
                        "type": "linear_gradient",
                        "geometry": {"x1": float("nan"), "y1": 0, "x2": 1, "y2": 1},
                        "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}],
                    },
                }
            ],
        }
    )
    with pytest.raises(PlanValidationError, match="/ops/0/fill/geometry/x1"):
        validate_plan(non_finite, _trace(), _scene("r1"))


def test_plan_requires_each_base_target_once_in_z_order():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    omitted = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [{"op": "set_z_order", "targets": ["r1"]}],
        }
    )
    with pytest.raises(PlanValidationError, match="/ops/0/targets"):
        validate_plan(omitted, _trace(), _scene("r1", "r2"))

    repeated = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [{"op": "set_z_order", "targets": ["r1", "r1"]}],
        }
    )
    with pytest.raises(PlanValidationError, match="/ops/0/targets/1"):
        validate_plan(repeated, _trace(), _scene("r1", "r2"))


def test_z_order_uses_the_final_target_created_by_grouping():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "merge", "id": "g1", "regions": ["r1"]},
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    validate_plan(plan, _trace(), _scene("r1"))


def test_parse_plan_recursively_freezes_nested_path_op_sequences():
    from vectormark.drawing_plan import parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    path_ops = tuple(payload["ops"][0]["geometry"]["ops"])
    payload["ops"][0]["geometry"]["ops"] = path_ops

    plan = parse_plan(payload)
    path_ops[0]["id"] = "changed-after-parsing"

    assert plan.ops[0]["geometry"]["ops"][0]["id"] == "s1"
    validate_plan(plan, _trace(), _scene("r1"))


@pytest.mark.parametrize(
    ("path_ops", "pointer"),
    [
        (
            [
                {"op": "group", "id": "s1", "commands": ["r1.p0.c1"]},
                {"op": "fit", "target": "s1", "type": "line"},
                {"op": "group", "id": "s2", "commands": ["r1.p0.c2"]},
                {"op": "fit", "target": "s2", "type": "line"},
                {"op": "break", "target": "s1"},
            ],
            "/ops/0/geometry/ops/4/target",
        ),
        (
            [
                {"op": "group", "id": "s1", "commands": ["r1.p0.c1"]},
                {"op": "fit", "target": "s1", "type": "line"},
                {"op": "break", "target": "s1"},
                {"op": "break", "target": "s1"},
            ],
            "/ops/0/geometry/ops/3/target",
        ),
    ],
)
def test_path_break_must_target_the_latest_unbroken_fit(path_ops, pointer):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    payload["ops"][0]["geometry"]["ops"] = path_ops

    with pytest.raises(PlanValidationError, match=pointer):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))


@pytest.mark.parametrize(
    "png_b64",
    [
        "this is not base64!",
        base64.b64encode(b"not a PNG").decode("ascii"),
    ],
)
def test_raster_fill_requires_base64_encoded_png_data(png_b64):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {
                    "op": "set_fill",
                    "target": "r1",
                    "fill": {
                        "type": "raster",
                        "geometry": {"x": 0, "y": 0, "w": 1, "h": 1},
                        "png_b64": png_b64,
                    },
                }
            ],
        }
    )

    with pytest.raises(PlanValidationError, match="/ops/0/fill/png_b64"):
        validate_plan(plan, _trace(), _scene("r1"))


def test_path_geometry_requires_at_least_one_operation():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    payload["ops"][0]["geometry"]["ops"] = []

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops"):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))


def test_plan_requires_path_dependencies():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    path_ops = payload["ops"][0]["geometry"]["ops"]
    path_ops[1]["target"] = "missing"

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/1/target"):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))
