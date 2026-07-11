"""Deterministic scene construction for validated drawing refinement plans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .drawing_plan import DrawingPlan, validate_plan
from .drawing_trace import TraceResult
from .emit import render_svg_doc, resolve_fill, shape_to_svg
from .fit import Shape, fit_path, _fmt


@dataclass(frozen=True)
class RenderTarget:
    id: str
    source_regions: tuple[str, ...]
    shape: Shape
    fill: Fill
    z: int


@dataclass(frozen=True)
class DrawingScene:
    targets: tuple[RenderTarget, ...]
    svg: str
    report: Mapping[str, object]


def _render(trace: TraceResult, targets: list[RenderTarget]) -> DrawingScene:
    ordered = sorted(targets, key=lambda target: target.z)
    defs: list[str] = []
    body = [shape_to_svg(target.shape, resolve_fill(target.fill, defs), target.id) for target in ordered]
    report = MappingProxyType({"targets": tuple({"id": t.id, "source_regions": t.source_regions,
        "geometry": t.shape.kind, "fill": type(t.fill).__name__, "z": t.z} for t in ordered)})
    return DrawingScene(tuple(ordered), render_svg_doc(trace.width, trace.height, body, defs), report)


def root_scene(trace: TraceResult) -> DrawingScene:
    targets = [RenderTarget(region.id, (region.id,), Shape("path", {"d": region.trace_path.d,
        "fill_rule": region.trace_path.fill_rule}), FlatFill(region.color), index)
        for index, region in enumerate(trace.regions)]
    return _render(trace, targets)


def refine(trace: TraceResult, base: DrawingScene, plan: DrawingPlan) -> DrawingScene:
    validate_plan(plan, trace, base)
    targets = {target.id: target for target in base.targets}
    z_order: list[str] | None = None
    for raw in plan.ops:
        op = raw
        if op["op"] == "group":
            members = [targets.pop(region) for region in op["regions"]]
            source_regions = tuple(source for member in members for source in member.source_regions)
            targets[op["id"]] = RenderTarget(op["id"], source_regions, members[0].shape, members[0].fill, min(m.z for m in members))
        elif op["op"] == "set_geometry":
            target = targets[op["target"]]
            geometry = op["geometry"]
            if geometry["type"] == "path":
                target = RenderTarget(target.id, target.source_regions, _path_shape(trace, target, geometry), target.fill, target.z)
            else:
                target = RenderTarget(target.id, target.source_regions, _primitive_shape(trace, target, geometry["type"]), target.fill, target.z)
            targets[target.id] = target
        elif op["op"] == "set_fill":
            target = targets[op["target"]]
            targets[target.id] = RenderTarget(target.id, target.source_regions, target.shape, _fill(op["fill"]), target.z)
        elif op["op"] == "set_z_order":
            z_order = list(op["targets"])
    if z_order is not None:
        targets = {identifier: RenderTarget(t.id, t.source_regions, t.shape, t.fill, index)
            for index, identifier in enumerate(z_order) for t in [targets[identifier]]}
    return _render(trace, list(targets.values()))


def _primitive_shape(trace: TraceResult, target: RenderTarget, kind: str) -> Shape:
    regions = [region for region in trace.regions if region.id in target.source_regions]
    xs = [x for region in regions for x in __import__("numpy").nonzero(region.mask)[1]]
    ys = [y for region in regions for y in __import__("numpy").nonzero(region.mask)[0]]
    x, y, w, h = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if kind == "rect": return Shape("rect", {"x": x, "y": y, "w": w, "h": h})
    if kind == "ellipse": return Shape("ellipse", {"cx": x + w / 2, "cy": y + h / 2, "rx": w / 2, "ry": h / 2})
    if kind == "circle": return Shape("circle", {"cx": x + w / 2, "cy": y + h / 2, "r": min(w, h) / 2})
    return Shape("polygon", {"points": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]})


def _command_start(trace: TraceResult, command_id: str) -> tuple[float, float]:
    for region in trace.regions:
        commands = region.trace_path.commands
        for index, command in enumerate(commands):
            if command.id == command_id:
                previous = commands[index - 1]
                return (float(previous.values[-2]), float(previous.values[-1]))
    raise KeyError(command_id)


def _path_shape(trace: TraceResult, target: RenderTarget, geometry: Mapping[str, object]) -> Shape:
    command_map = {command.id: command for region in trace.regions for command in region.trace_path.commands}
    groups: dict[str, list[object]] = {}
    pieces: list[str] = []
    close = False
    for op in geometry["ops"]:
        if op["op"] == "group":
            groups[op["id"]] = [command_map[command_id] for command_id in op["commands"]]
        elif op["op"] == "fit":
            commands = groups[op["target"]]
            start = _command_start(trace, commands[0].id)
            points = np.array([start] + [(command.values[-2], command.values[-1]) for command in commands], dtype=float)
            kind = op["type"]
            if kind == "line":
                pieces.append(f"M{_fmt(start[0])} {_fmt(start[1])} L{_fmt(points[-1, 0])} {_fmt(points[-1, 1])}")
            elif kind == "keep":
                body = " ".join(command.command + " ".join(_fmt(value) for value in command.values) for command in commands)
                pieces.append(f"M{_fmt(start[0])} {_fmt(start[1])} {body}")
            else:
                pieces.append(str(fit_path(points, epsilon=0.0, max_error=1.0, cubic=kind == "cubic").params["d"]))
        elif op["op"] == "close":
            close = True
    d = " ".join(pieces)
    if close and not d.endswith("Z"):
        d += " Z"
    return Shape("path", {"d": d})


def _fill(spec: Mapping[str, object]) -> Fill:
    kind = spec["type"]
    if kind == "flat": return FlatFill(spec["color"])
    if kind == "linear_gradient": return LinearGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    if kind == "radial_gradient": return RadialGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    return RasterFill(dict(spec["geometry"]), spec["png_b64"])
