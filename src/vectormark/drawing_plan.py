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

from .drawing_trace import TraceCommand, TraceResult


_VERSION = "vectormark.plan.v1"
_PLAN_KEYS = frozenset({"version", "drawing_id", "base_version", "label", "ops"})
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_BASE_VERSION = re.compile(r"^v\d+(?:\.\d+)*$")
_GEOMETRY_TYPES = frozenset({"circle", "ellipse", "rect", "polygon", "path"})
_FILL_TYPES = frozenset({"flat", "linear_gradient", "radial_gradient", "raster"})
_FIT_TYPES = frozenset({"line", "quadratic", "cubic", "keep"})
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
    ops = payload.get("ops")
    if not isinstance(ops, list):
        raise PlanValidationError("/ops", "must be an array")
    frozen_ops = tuple(_freeze(op) for op in ops)
    for index, op in enumerate(frozen_ops):
        if not isinstance(op, Mapping):
            raise PlanValidationError(f"/ops/{index}", "must be an object")
    return DrawingPlan(_VERSION, drawing_id, base_version, label, frozen_ops)


def validate_plan(plan: DrawingPlan, trace: TraceResult, scene: object) -> None:
    """Validate plan references and semantics against one trace and base scene."""
    target_ids = _scene_target_ids(scene)
    trace_region_ids = {region.id for region in trace.regions}
    commands = _trace_commands(trace)
    known_targets = set(target_ids)
    used_target_ids = set(target_ids)
    z_order_seen = False
    z_order: tuple[Mapping[object, object], str] | None = None

    for index, raw_op in enumerate(plan.ops):
        pointer = f"/ops/{index}"
        op = _mapping(raw_op, pointer)
        name = _required_str(op, "op", pointer)
        if name == "group":
            _reject_unknown_keys(op, {"op", "id", "regions"}, pointer)
            group_id = _required_str(op, "id", pointer)
            if group_id in used_target_ids:
                raise PlanValidationError(f"{pointer}/id", "duplicate operation id")
            regions = _strings(op.get("regions"), f"{pointer}/regions", nonempty=True)
            for region_index, region_id in enumerate(regions):
                if region_id not in trace_region_ids or region_id not in known_targets:
                    raise PlanValidationError(f"{pointer}/regions/{region_index}", "unknown region")
            if len(set(regions)) != len(regions):
                repeated = _first_repeat(regions)
                raise PlanValidationError(f"{pointer}/regions/{repeated}", "region is repeated")
            known_targets.difference_update(regions)
            known_targets.add(group_id)
            used_target_ids.add(group_id)
        elif name == "set_geometry":
            _validate_geometry_op(op, pointer, known_targets, commands)
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
    op: Mapping[object, object], pointer: str, targets: set[str], commands: Mapping[str, tuple[int, int, TraceCommand]],
) -> None:
    _reject_unknown_keys(op, {"op", "target", "geometry"}, pointer)
    _validate_target(op, pointer, targets)
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
    _validate_path_ops(path_ops, f"{pointer}/geometry/ops", commands)


def _validate_path_ops(
    path_ops: Sequence[object], pointer: str, commands: Mapping[str, tuple[int, int, TraceCommand]],
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
            records: list[tuple[int, int, TraceCommand]] = []
            for command_index, command_id in enumerate(selected):
                try:
                    record = commands[command_id]
                except KeyError:
                    raise PlanValidationError(f"{op_pointer}/commands/{command_index}", "unknown trace command") from None
                if record[2].command in {"M", "Z"}:
                    raise PlanValidationError(f"{op_pointer}/commands/{command_index}", "M and Z commands cannot be grouped")
                records.append(record)
            for command_index in range(1, len(records)):
                previous, current = records[command_index - 1], records[command_index]
                if current[:2] != (previous[0], previous[1] + 1):
                    raise PlanValidationError(
                        f"{op_pointer}/commands/{command_index}",
                        "commands must be a contiguous run from one trace subpath",
                    )
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


def _scene_target_ids(scene: object) -> tuple[str, ...]:
    targets = getattr(scene, "targets", None)
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise PlanValidationError("/ops", "base scene must expose targets")
    ids: list[str] = []
    for target in targets:
        target_id = getattr(target, "id", None)
        if type(target_id) is not str:
            raise PlanValidationError("/ops", "base scene target ids must be strings")
        ids.append(target_id)
    if len(set(ids)) != len(ids):
        raise PlanValidationError("/ops", "base scene contains duplicate target ids")
    return tuple(ids)


def _trace_commands(trace: TraceResult) -> dict[str, tuple[int, int, TraceCommand]]:
    commands: dict[str, tuple[int, int, TraceCommand]] = {}
    for region in trace.regions:
        subpath = -1
        index = 0
        for command in region.trace_path.commands:
            if command.command == "M":
                subpath += 1
                index = 0
            if command.id in commands:
                raise PlanValidationError("/ops", "trace contains duplicate command ids")
            commands[command.id] = (subpath, index, command)
            index += 1
    return commands


def _validate_target(op: Mapping[object, object], pointer: str, targets: set[str]) -> None:
    target = _required_str(op, "target", pointer)
    if target not in targets:
        raise PlanValidationError(f"{pointer}/target", "unknown target")


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
