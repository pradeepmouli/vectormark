from types import SimpleNamespace

import numpy as np
import pytest

from vectormark.drawing_trace import (
    TraceCommand,
    TraceOptions,
    TracePath,
    TraceRegion,
    TraceResult,
)


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


def _scene(*target_ids: str) -> object:
    return SimpleNamespace(targets=tuple(SimpleNamespace(id=target_id) for target_id in target_ids))


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


def test_plan_reports_pointer_for_non_contiguous_path_commands():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(_path_plan(["r1.p0.c1", "r1.p0.c3"]))

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/0/commands/1"):
        validate_plan(plan, _trace(), _scene("r1"))


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
                    {"op": "group", "id": "g1", "regions": ["r1"]},
                    {"op": "group", "id": "g1", "regions": ["r1"]},
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


def test_plan_requires_path_dependencies():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan(["r1.p0.c1"])
    path_ops = payload["ops"][0]["geometry"]["ops"]
    path_ops[1]["target"] = "missing"

    with pytest.raises(PlanValidationError, match="/ops/0/geometry/ops/1/target"):
        validate_plan(parse_plan(payload), _trace(), _scene("r1"))
