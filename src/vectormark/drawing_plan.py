"""Typed parsing and semantic validation for drawing refinement plans."""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .drawing_trace import TraceCommand, TraceResult, svg_path_commands
from .optimizer.vector_region import VectorRegion


_VERSION = "vectormark.plan.v1"
_PLAN_KEYS = frozenset({"version", "drawing_id", "base_version", "label", "defaults", "ops"})
_TOLERANCE_KEYS = frozenset({"epsilon", "max_error"})
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_BASE_VERSION = re.compile(r"^v\d+(?:\.\d+)*$")
_GEOMETRY_TYPES = frozenset(
    {"circle", "ellipse", "rect", "rounded_rect", "polygon", "trapezoid", "rounded_trapezoid", "cap", "path"}
)
_FILL_TYPES = frozenset({"flat", "linear_gradient", "radial_gradient", "raster"})
_FIT_TYPES = frozenset({"line", "quadratic", "cubic", "keep"})
_DETECT_OPS = frozenset({"detect_primitives", "detect_symmetry", "detect_clones"})
_POLISH_OPS = frozenset({"simplify", "stitch"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PlanValidationError(ValueError):
    """A user-facing plan error located by an RFC 6901 JSON pointer."""

    def __init__(self, pointer: str, message: str) -> None:
        self.pointer = pointer
        self.message = message
        super().__init__(f"{pointer}: {message}")


@dataclass(frozen=True)
class DrawingPlan:
    version: Literal["vectormark.plan.v1"]
    drawing_id: str
    base_version: str
    label: str | None
    defaults: Mapping[str, float]
    ops: tuple[object, ...]


def parse_plan(payload: Mapping[str, object]) -> DrawingPlan:
    """Parse a JSON-ready plan and detach it from the caller's mutable payload."""
    if not isinstance(payload, Mapping):
        raise PlanValidationError("", "plan must be an object")
    for key in payload:
        if key not in _PLAN_KEYS:
            raise PlanValidationError(f"/{_escape(key)}", "unknown plan field")

    version = _required_str(payload, "version", "")
    if version != _VERSION:
        raise PlanValidationError("/version", f"expected {_VERSION!r}")
    drawing_id = _required_str(payload, "drawing_id", "")
    if not drawing_id:
        raise PlanValidationError("/drawing_id", "must not be empty")
    base_version = _required_str(payload, "base_version", "")
    if not _BASE_VERSION.fullmatch(base_version):
        raise PlanValidationError("/base_version", "must be a version such as 'v0' or 'v0.1'")
    label = payload.get("label")
    if type(label) not in (str, type(None)):
        raise PlanValidationError("/label", "must be a string or null")
    if isinstance(label, str) and len(label) > 200:
        raise PlanValidationError("/label", "must be at most 200 characters")
    defaults_value = _mapping(payload.get("defaults", {}), "/defaults")
    _reject_unknown_keys(defaults_value, set(_TOLERANCE_KEYS), "/defaults")
    defaults = _tolerances(defaults_value, "/defaults")
    ops = payload.get("ops")
    if not isinstance(ops, list):
        raise PlanValidationError("/ops", "must be an array")
    frozen_ops = tuple(_freeze(op) for op in ops)
    for index, op in enumerate(frozen_ops):
        if not isinstance(op, Mapping):
            raise PlanValidationError(f"/ops/{index}", "must be an object")
    return DrawingPlan(_VERSION, drawing_id, base_version, label, MappingProxyType(defaults), frozen_ops)


def validate_plan(plan: DrawingPlan, trace: TraceResult, regions: Sequence[VectorRegion]) -> None:
    """Validate plan references and semantics against one trace and region roots."""
    source_regions = _region_target_source_regions(regions)
    target_ids = tuple(source_regions)
    commands = _target_commands(regions)
    known_targets = set(target_ids)
    used_commands: set[str] = set()
    used_target_ids = set(target_ids)
    z_order_seen = False
    z_order: tuple[Mapping[object, object], str] | None = None
    symmetry_targets: set[str] = set()
    symmetry_block = False

    for index, raw_op in enumerate(plan.ops):
        pointer = f"/ops/{index}"
        op = _mapping(raw_op, pointer)
        name = _required_str(op, "op", pointer)
        if symmetry_block and name != "detect_symmetry":
            raise PlanValidationError(pointer, "detect_symmetry must be the final operation or part of a terminal targeted-symmetry block")
        if name == "merge":
            _reject_unknown_keys(op, {"op", "id", "regions"}, pointer)
            group_id = _required_str(op, "id", pointer)
            if group_id in used_target_ids:
                raise PlanValidationError(f"{pointer}/id", "duplicate operation id")
            regions = _strings(op.get("regions"), f"{pointer}/regions", nonempty=True)
            for region_index, region_id in enumerate(regions):
                if region_id not in known_targets:
                    raise PlanValidationError(f"{pointer}/regions/{region_index}", "unknown region")
            if len(set(regions)) != len(regions):
                repeated = _first_repeat(regions)
                raise PlanValidationError(f"{pointer}/regions/{repeated}", "region is repeated")
            group_sources = frozenset().union(*(source_regions[region_id] for region_id in regions))
            known_targets.difference_update(regions)
            known_targets.add(group_id)
            for region_id in regions:
                del source_regions[region_id]
            source_regions[group_id] = group_sources
            used_target_ids.add(group_id)
        elif name == "split":
            _reject_unknown_keys(op, {"op", "target"}, pointer)
            _validate_target(op, pointer, known_targets)
            if index != len(plan.ops) - 1:
                raise PlanValidationError(pointer, "split must be the final operation; refine child regions in a new version")
        elif name in _DETECT_OPS:
            _validate_detect_op(op, pointer, known_targets)
            if name == "detect_symmetry":
                target = op.get("target")
                if target is None:
                    if index != len(plan.ops) - 1:
                        raise PlanValidationError(pointer, "global detect_symmetry must be the final operation; refine derived child regions in a new version")
                    if symmetry_block:
                        raise PlanValidationError(pointer, "global detect_symmetry cannot follow targeted symmetry operations")
                else:
                    assert type(target) is str
                    if target in symmetry_targets:
                        raise PlanValidationError(f"{pointer}/target", "target already has a symmetry operation in this plan")
                    symmetry_targets.add(target)
                    symmetry_block = True
        elif name in _POLISH_OPS:
            _validate_polish_op(op, pointer, known_targets)
        elif name == "set_symmetry":
            _validate_set_symmetry_op(op, pointer, known_targets)
        elif name == "clone":
            _validate_clone_op(op, pointer, known_targets)
        elif name == "set_geometry":
            _validate_geometry_op(op, pointer, known_targets, commands, used_commands)
        elif name == "set_fill":
            _validate_fill_op(op, pointer, known_targets)
        elif name == "set_z_order":
            if z_order_seen:
                raise PlanValidationError(pointer, "only one set_z_order operation is allowed")
            z_order_seen = True
            z_order = (op, pointer)
        else:
            raise PlanValidationError(f"{pointer}/op", "unsupported operation")

    if z_order is not None:
        _validate_z_order(*z_order, known_targets)


def _validate_geometry_op(
    op: Mapping[object, object],
    pointer: str,
    targets: set[str],
    commands: Mapping[str, tuple[str, int, int, TraceCommand]],
    used_commands: set[str],
) -> None:
    _reject_unknown_keys(op, {"op", "target", "geometry"} | _TOLERANCE_KEYS, pointer)
    _tolerances(op, pointer)
    target = _validate_target(op, pointer, targets)
    geometry = _mapping(op.get("geometry"), f"{pointer}/geometry")
    geometry_type = _required_str(geometry, "type", f"{pointer}/geometry")
    if geometry_type not in _GEOMETRY_TYPES:
        raise PlanValidationError(f"{pointer}/geometry/type", "unsupported geometry type")
    if geometry_type != "path":
        _reject_unknown_keys(geometry, {"type"}, f"{pointer}/geometry")
        return
    _reject_unknown_keys(geometry, {"type", "ops"}, f"{pointer}/geometry")
    path_ops = geometry.get("ops")
    if not isinstance(path_ops, Sequence) or isinstance(path_ops, (str, bytes)):
        raise PlanValidationError(f"{pointer}/geometry/ops", "must be an array")
    if not path_ops:
        raise PlanValidationError(f"{pointer}/geometry/ops", "must not be empty")
    _validate_path_ops(
        path_ops,
        f"{pointer}/geometry/ops",
        commands,
        target,
        used_commands,
    )


def _validate_detect_op(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "target"} | _TOLERANCE_KEYS, pointer)
    _tolerances(op, pointer)
    if "target" in op:
        _validate_target(op, pointer, targets)


def _validate_polish_op(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "target"} | _TOLERANCE_KEYS, pointer)
    _tolerances(op, pointer)
    if "target" in op:
        _validate_target(op, pointer, targets)


def _validate_set_symmetry_op(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "source", "target", "axis"}, pointer)
    source = _required_str(op, "source", pointer)
    target = _required_str(op, "target", pointer)
    if source not in targets:
        raise PlanValidationError(f"{pointer}/source", "unknown target")
    if target not in targets:
        raise PlanValidationError(f"{pointer}/target", "unknown target")
    if source == target:
        raise PlanValidationError(f"{pointer}/target", "must differ from source")
    _validate_axis(op.get("axis"), f"{pointer}/axis")


def _validate_clone_op(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "source", "target", "transform"}, pointer)
    source = _required_str(op, "source", pointer)
    target = _required_str(op, "target", pointer)
    if source not in targets:
        raise PlanValidationError(f"{pointer}/source", "unknown target")
    if target not in targets:
        raise PlanValidationError(f"{pointer}/target", "unknown target")
    if source == target:
        raise PlanValidationError(f"{pointer}/target", "must differ from source")
    transform = op.get("transform")
    if not isinstance(transform, Sequence) or isinstance(transform, (str, bytes)) or len(transform) != 6:
        raise PlanValidationError(f"{pointer}/transform", "must be a six-number affine matrix")
    for index, value in enumerate(transform):
        if not _finite_number(value):
            raise PlanValidationError(f"{pointer}/transform/{index}", "must be a finite number")


def _validate_axis(value: object, pointer: str) -> None:
    axis = _mapping(value, pointer)
    _reject_unknown_keys(axis, {"theta", "cx", "cy"}, pointer)
    for key in ("theta", "cx", "cy"):
        if not _finite_number(axis.get(key)):
            raise PlanValidationError(f"{pointer}/{key}", "must be a finite number")


def _validate_path_ops(
    path_ops: Sequence[object],
    pointer: str,
    commands: Mapping[str, tuple[str, int, int, TraceCommand]],
    target_id: str,
    used_commands: set[str],
) -> None:
    groups: set[str] = set()
    fitted: set[str] = set()
    since_close = False
    latest_fitted: str | None = None
    broken: set[str] = set()
    for index, raw_op in enumerate(path_ops):
        op_pointer = f"{pointer}/{index}"
        op = _mapping(raw_op, op_pointer)
        name = _required_str(op, "op", op_pointer)
        if name == "group":
            _reject_unknown_keys(op, {"op", "id", "commands"}, op_pointer)
            group_id = _required_str(op, "id", op_pointer)
            if group_id in groups:
                raise PlanValidationError(f"{op_pointer}/id", "duplicate path group id")
            selected = _strings(op.get("commands"), f"{op_pointer}/commands", nonempty=True)
            records: list[tuple[str, int, int, TraceCommand]] = []
            for command_index, command_id in enumerate(selected):
                try:
                    record = commands[command_id]
                except KeyError:
                    raise PlanValidationError(f"{op_pointer}/commands/{command_index}", "unknown retained-path command") from None
                if record[0] != target_id:
                    raise PlanValidationError(
                        f"{op_pointer}/commands/{command_index}", "command is not owned by the target's retained path"
                    )
                if command_id in used_commands:
                    raise PlanValidationError(
                        f"{op_pointer}/commands/{command_index}", "trace command is already used by another path group"
                    )
                if record[3].command in {"M", "Z"}:
                    raise PlanValidationError(f"{op_pointer}/commands/{command_index}", "M and Z commands cannot be grouped")
                records.append(record)
            for command_index in range(1, len(records)):
                previous, current = records[command_index - 1], records[command_index]
                if current[:3] != (previous[0], previous[1], previous[2] + 1):
                    raise PlanValidationError(
                        f"{op_pointer}/commands/{command_index}",
                        "commands must be a contiguous run from one trace subpath",
                    )
            used_commands.update(selected)
            groups.add(group_id)
        elif name == "fit":
            _reject_unknown_keys(op, {"op", "target", "type"}, op_pointer)
            target = _required_str(op, "target", op_pointer)
            if target not in groups:
                raise PlanValidationError(f"{op_pointer}/target", "unknown or not-yet-defined path group")
            if target in fitted:
                raise PlanValidationError(f"{op_pointer}/target", "path group is already fitted")
            fit_type = _required_str(op, "type", op_pointer)
            if fit_type not in _FIT_TYPES:
                raise PlanValidationError(f"{op_pointer}/type", "unsupported fit type")
            fitted.add(target)
            since_close = True
            latest_fitted = target
        elif name == "break":
            _reject_unknown_keys(op, {"op", "target"}, op_pointer)
            target = _required_str(op, "target", op_pointer)
            if target != latest_fitted or target in broken:
                raise PlanValidationError(f"{op_pointer}/target", "break requires the current unbroken fitted path group")
            broken.add(target)
        elif name == "close":
            _reject_unknown_keys(op, {"op"}, op_pointer)
            if not since_close:
                raise PlanValidationError(op_pointer, "close requires a preceding fit")
            since_close = False
            latest_fitted = None
        elif name in {"simplify", "stitch"}:
            _reject_unknown_keys(op, {"op"}, op_pointer)
            if not fitted:
                raise PlanValidationError(op_pointer, f"{name} requires a preceding fit")
        else:
            raise PlanValidationError(f"{op_pointer}/op", "unsupported path operation")
    if groups - fitted:
        missing = next(index for index, raw_op in enumerate(path_ops) if _path_group_id(raw_op) in groups - fitted)
        raise PlanValidationError(f"{pointer}/{missing}/id", "path group requires a fit operation")


def _validate_fill_op(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "target", "fill"}, pointer)
    _validate_target(op, pointer, targets)
    fill = _mapping(op.get("fill"), f"{pointer}/fill")
    fill_type = _required_str(fill, "type", f"{pointer}/fill")
    if fill_type not in _FILL_TYPES:
        raise PlanValidationError(f"{pointer}/fill/type", "unsupported fill type")
    if fill_type == "flat":
        _reject_unknown_keys(fill, {"type", "color"}, f"{pointer}/fill")
        _color(fill.get("color"), f"{pointer}/fill/color")
        return
    geometry_keys = {"x1", "y1", "x2", "y2"} if fill_type == "linear_gradient" else {"cx", "cy", "r"} if fill_type == "radial_gradient" else {"x", "y", "w", "h"}
    allowed = {"type", "geometry"} | ({"stops"} if fill_type != "raster" else {"png_b64"})
    _reject_unknown_keys(fill, allowed, f"{pointer}/fill")
    geometry = _mapping(fill.get("geometry"), f"{pointer}/fill/geometry")
    _reject_unknown_keys(geometry, geometry_keys, f"{pointer}/fill/geometry")
    for key in geometry_keys:
        value = geometry.get(key)
        if not _finite_number(value):
            raise PlanValidationError(f"{pointer}/fill/geometry/{key}", "must be a finite number")
    if fill_type in {"radial_gradient", "raster"}:
        positive_key = "r" if fill_type == "radial_gradient" else "w"
        if float(geometry[positive_key]) <= 0:
            raise PlanValidationError(f"{pointer}/fill/geometry/{positive_key}", "must be greater than zero")
    if fill_type == "raster":
        if float(geometry["h"]) <= 0:
            raise PlanValidationError(f"{pointer}/fill/geometry/h", "must be greater than zero")
        if type(fill.get("png_b64")) is not str or not fill["png_b64"]:
            raise PlanValidationError(f"{pointer}/fill/png_b64", "must be a non-empty base64 string")
        try:
            png_data = base64.b64decode(fill["png_b64"], validate=True)
        except (binascii.Error, ValueError):
            raise PlanValidationError(f"{pointer}/fill/png_b64", "must be valid base64 PNG data") from None
        if not png_data.startswith(_PNG_SIGNATURE):
            raise PlanValidationError(f"{pointer}/fill/png_b64", "must contain PNG data")
        return
    stops = fill.get("stops")
    if not isinstance(stops, Sequence) or isinstance(stops, (str, bytes)) or len(stops) < 2:
        raise PlanValidationError(f"{pointer}/fill/stops", "must contain at least two stops")
    for index, raw_stop in enumerate(stops):
        stop_pointer = f"{pointer}/fill/stops/{index}"
        stop = _mapping(raw_stop, stop_pointer)
        _reject_unknown_keys(stop, {"offset", "color"}, stop_pointer)
        offset = stop.get("offset")
        if not _finite_number(offset) or not 0 <= float(offset) <= 1:
            raise PlanValidationError(f"{stop_pointer}/offset", "must be a finite number between zero and one")
        _color(stop.get("color"), f"{stop_pointer}/color")


def _validate_z_order(op: Mapping[object, object], pointer: str, target_ids: set[str]) -> None:
    _reject_unknown_keys(op, {"op", "targets"}, pointer)
    targets = _strings(op.get("targets"), f"{pointer}/targets", nonempty=True)
    seen: set[str] = set()
    for index, target in enumerate(targets):
        if target in seen:
            raise PlanValidationError(f"{pointer}/targets/{index}", "target is repeated")
        seen.add(target)
        if target not in target_ids:
            raise PlanValidationError(f"{pointer}/targets/{index}", "unknown target")
    if set(targets) != target_ids:
        raise PlanValidationError(f"{pointer}/targets", "must list every target exactly once")


def _region_target_source_regions(regions: Sequence[VectorRegion]) -> dict[str, frozenset[str]]:
    if isinstance(regions, (str, bytes)):
        raise PlanValidationError("/ops", "base drawing must expose vector regions")
    source_regions: dict[str, frozenset[str]] = {}

    def visit(region: VectorRegion, target_id: str, sources: frozenset[str]) -> None:
        if region.is_leaf:
            if target_id in source_regions:
                raise PlanValidationError("/ops", "base drawing contains duplicate target ids")
            source_regions[target_id] = sources
            return
        for child in region.children:
            visit(child, f"{target_id}-{child.id}", sources)

    for region in regions:
        if type(region.drawing_id) is not str:
            raise PlanValidationError("/ops", "base drawing regions require stable drawing ids")
        visit(region, region.drawing_id, frozenset(region.source_regions or (region.drawing_id,)))
    return source_regions


def _target_commands(regions: Sequence[VectorRegion]) -> dict[str, tuple[str, int, int, TraceCommand]]:
    commands: dict[str, tuple[str, int, int, TraceCommand]] = {}

    def visit(region: VectorRegion, target_id: str) -> None:
        if region.is_leaf:
            assert region.current is not None
            if region.current.kind != "path":
                return
            subpath = -1
            index = 0
            for command in svg_path_commands(str(region.current.params["d"]), target_id):
                if command.command == "M":
                    subpath += 1
                    index = 0
                if command.id in commands:
                    raise PlanValidationError("/ops", "retained drawing contains duplicate path command ids")
                commands[command.id] = (target_id, subpath, index, command)
                index += 1
            return
        for child in region.children:
            visit(child, f"{target_id}-{child.id}")

    for region in regions:
        if type(region.drawing_id) is not str:
            raise PlanValidationError("/ops", "base drawing regions require stable drawing ids")
        visit(region, region.drawing_id)
    return commands


def _validate_target(op: Mapping[object, object], pointer: str, targets: set[str]) -> str:
    target = _required_str(op, "target", pointer)
    if target not in targets:
        raise PlanValidationError(f"{pointer}/target", "unknown target")
    return target


def _required_str(mapping: Mapping[object, object], key: str, pointer: str) -> str:
    value = mapping.get(key)
    if type(value) is not str:
        raise PlanValidationError(f"{pointer}/{key}" if pointer else f"/{key}", "must be a string")
    return value


def _mapping(value: object, pointer: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise PlanValidationError(pointer, "must be an object")
    return value


def _strings(value: object, pointer: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlanValidationError(pointer, "must be an array")
    if nonempty and not value:
        raise PlanValidationError(pointer, "must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise PlanValidationError(f"{pointer}/{index}", "must be a string")
        result.append(item)
    return tuple(result)


def _reject_unknown_keys(mapping: Mapping[object, object], allowed: set[str], pointer: str) -> None:
    for key in mapping:
        if type(key) is not str or key not in allowed:
            raise PlanValidationError(f"{pointer}/{_escape(str(key))}", "unknown field")


def _color(value: object, pointer: str) -> None:
    if type(value) is not str or not _HEX_COLOR.fullmatch(value):
        raise PlanValidationError(pointer, "must be a #RRGGBB color")


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _tolerances(value: object, pointer: str) -> dict[str, float]:
    """Validate optional fitting tolerances at plan or operation scope."""
    tolerances = _mapping(value, pointer)
    result: dict[str, float] = {}
    for key in _TOLERANCE_KEYS:
        if key not in tolerances:
            continue
        raw = tolerances[key]
        if not _finite_number(raw) or float(raw) < 0:
            raise PlanValidationError(f"{pointer}/{key}", "must be a finite number greater than or equal to zero")
        result[key] = float(raw)
    return result


def _first_repeat(items: Sequence[str]) -> int:
    seen: set[str] = set()
    for index, item in enumerate(items):
        if item in seen:
            return index
        seen.add(item)
    raise AssertionError("expected a repeated item")


def _path_group_id(value: object) -> str | None:
    if not isinstance(value, Mapping) or value.get("op") != "group":
        return None
    group_id = value.get("id")
    return group_id if type(group_id) is str else None


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _escape(token: object) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")
