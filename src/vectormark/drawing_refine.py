"""Apply drawing plans directly to cached :class:`VectorRegion` roots.

The interactive drawing state is the optimizer's native region forest.  SVG and
reports are derived artifacts, never a second editable scene representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .drawing_plan import DrawingPlan, validate_plan
from .drawing_trace import TraceResult
from .emit import render_svg_doc, resolve_fill, resolve_use_shape, shape_to_svg
from .fit import Shape, _fmt, fit_path
from .optimizer.vector_region import VectorRegion
from .optimizer.framework import optimize
from .optimizer.passes import clones_pass, primitives_pass, seams_pass, simplify_pass, split_compound_pass, symmetry_pass
from .optimizer.shape_transform import bake_shape_transform
from .skia_geometry import SkPath, affinity


DrawingRegions = tuple[VectorRegion, ...]


@dataclass(frozen=True)
class RenderedDrawing:
    """A transient SVG/report projection of one drawing-version's regions."""

    svg: str
    report: Mapping[str, object]


@dataclass(frozen=True)
class _Target:
    id: str
    region: VectorRegion
    source_regions: frozenset[str]


def root_regions(trace: TraceResult) -> DrawingRegions:
    """Create the native region roots retained for an interactive trace."""
    return tuple(
        VectorRegion.from_shape(
            id=index + 1,
            shape=Shape("path", {"d": region.trace_path.d, "fill_rule": region.trace_path.fill_rule}),
            fill=FlatFill(region.color),
            z=index,
            raster=region.mask,
            source_label=region.source_label,
            color_hex=region.color,
            drawing_id=region.id,
            source_regions=(region.id,),
        )
        for index, region in enumerate(trace.regions)
    )


def render_drawing(trace: TraceResult, regions: Sequence[VectorRegion]) -> RenderedDrawing:
    """Render cached regions without changing the drawing-version state."""
    targets = _targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    id_map = {target.region.id: target.id for target in ordered}
    defs: list[str] = []
    body: list[str] = []
    report_targets: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        shape = resolve_use_shape(target.region.current, id_map)
        fill = target.region.fill
        if fill is None:
            raise ValueError(f"drawing region {target.id!r} has no fill")
        body.append(shape_to_svg(shape, resolve_fill(fill, defs), target.id))
        report_targets.append(
            {
                "id": target.id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": shape.kind,
                "fill": type(fill).__name__,
                "z": target.region.z,
            }
        )
    return RenderedDrawing(
        render_svg_doc(trace.width, trace.height, body, defs),
        {"targets": tuple(report_targets)},
    )


def refine(trace: TraceResult, base: Sequence[VectorRegion], plan: DrawingPlan) -> DrawingRegions:
    """Execute *plan* as immutable transformations over native vector regions."""
    regions = tuple(base)
    validate_plan(plan, trace, regions)
    z_order: tuple[str, ...] | None = None

    for raw in plan.ops:
        op = raw
        name = op["op"]
        if name == "merge":
            regions = _group_regions(regions, op)
        elif name == "split":
            regions = _run_detection(regions, trace, "split", op["target"])
        elif name in {"detect_primitives", "detect_symmetry", "detect_clones"}:
            regions = _run_detection(regions, trace, name, op.get("target"))
        elif name == "set_symmetry":
            regions = _set_symmetry(regions, op)
        elif name == "clone":
            regions = _clone(regions, op)
        elif name == "set_geometry":
            target = _targets(regions)[op["target"]]
            geometry = op["geometry"]
            shape = (
                _path_shape(trace, target, geometry)
                if geometry["type"] == "path"
                else _primitive_shape(trace, target, geometry["type"])
            )
            regions = _replace_leaf(regions, target.id, lambda region: region.with_current(shape))
            path_ops = geometry.get("ops", ()) if geometry["type"] == "path" else ()
            if any(path_op["op"] == "simplify" for path_op in path_ops):
                regions = _run_detection(regions, trace, "simplify", target.id)
            if any(path_op["op"] == "seams" for path_op in path_ops):
                regions = _run_detection(regions, trace, "seams", target.id)
        elif name == "set_fill":
            target_id = op["target"]
            fill = _fill(op["fill"])
            regions = _replace_leaf(regions, target_id, lambda region: _with_fill(region, fill))
        elif name == "set_z_order":
            z_order = tuple(op["targets"])

    if z_order is not None:
        for z, target_id in enumerate(z_order):
            regions = _replace_leaf(regions, target_id, lambda region, z=z: _with_z(region, z))
    return regions


def _run_detection(
    regions: Sequence[VectorRegion],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
) -> DrawingRegions:
    """Run one existing optimizer pass over the cached region forest."""
    epsilon = trace.options.simplify_tolerance
    max_error = trace.options.curve_tolerance
    cubic = trace.options.curve_type == "cubic"
    if operation == "detect_primitives":
        def pass_fn(objects, masks):
            return primitives_pass(objects, masks, epsilon=epsilon)
    elif operation == "detect_symmetry":
        def pass_fn(objects, masks):
            return symmetry_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    elif operation == "detect_clones":
        pass_fn = clones_pass
    elif operation == "split":
        pass_fn = split_compound_pass
    elif operation == "simplify":
        def pass_fn(objects, masks):
            return simplify_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    elif operation == "seams":
        def pass_fn(objects, masks):
            return seams_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    else:
        raise ValueError(f"unknown detection operation: {operation}")

    selected_id = target_id if type(target_id) is str else None
    selected_region_id: int | None = None
    if selected_id is not None:
        selected_region_id = _targets(regions)[selected_id].region.id
        original_pass = pass_fn

        def pass_fn(objects, masks):  # type: ignore[no-redef]
            return [proposal for proposal in original_pass(objects, masks) if selected_region_id in proposal.obj_ids]

    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    optimized = optimize(list(regions), masks, [pass_fn])
    return tuple(_restore_root_metadata(region, before.get(region.id)) for region in optimized)


def _restore_root_metadata(region: VectorRegion, original: VectorRegion | None) -> VectorRegion:
    """Optimizer passes preserve geometry IDs, while drawing IDs stay root-owned."""
    if original is None or original.drawing_id is None:
        return region
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
            id=region.id,
            current=region.current,
            original=region.original,
            fill=region.fill,
            z=region.z,
            footprint=region.footprint,
            raster=region.raster,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=original.drawing_id,
            source_regions=original.source_regions,
            coverage=region.coverage,
            diagnostics=region.diagnostics,
        )
    return VectorRegion.branch(
        id=region.id,
        children=region.children,
        z=region.z,
        raster=region.raster,
        footprint=region.footprint,
        fill=region.fill,
        source_label=region.source_label,
        color_hex=region.color_hex,
        drawing_id=original.drawing_id,
        source_regions=original.source_regions,
        diagnostics=region.diagnostics,
    )


def _clone(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    targets = _targets(regions)
    source = targets[op["source"]].region
    target = targets[op["target"]].region
    assert source.current is not None and source.fill is not None and target.current is not None
    transform = tuple(float(value) for value in op["transform"])
    shape = Shape("use", {"href_obj_id": source.id, "transform": transform})
    footprint = _transform_footprint(source.footprint, transform, target.footprint)
    return _replace_leaf(
        regions,
        op["target"],
        lambda region: region.with_current(
            shape,
            fill=source.fill,
            footprint=footprint,
            diagnostics={"clone": {"source_id": source.id, "transform": transform}},
        ),
    )


def _transform_footprint(
    footprint: object,
    transform: tuple[float, float, float, float, float, float],
    fallback: object,
) -> object:
    if not isinstance(footprint, SkPath):
        return fallback
    a, b, c, d, e, f = transform
    return affinity.affine_transform(footprint, [a, c, b, d, e, f])


def _set_symmetry(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Set a mirror-pair relationship from an agent-supplied reflection axis."""
    targets = _targets(regions)
    source = targets[op["source"]].region
    assert source.current is not None and source.fill is not None
    axis = op["axis"]
    matrix = _reflection_matrix(float(axis["theta"]), float(axis["cx"]), float(axis["cy"]))
    reflected = bake_shape_transform(source.current, matrix)
    return _replace_leaf(
        regions,
        op["target"],
        lambda region: region.with_current(
            reflected,
            fill=source.fill,
            diagnostics={"symmetry": {"mode": "pair", "source_id": source.id, "axis": dict(axis)}},
        ),
    )


def _reflection_matrix(theta: float, cx: float, cy: float) -> tuple[float, float, float, float, float, float]:
    ux, uy = math.cos(theta), math.sin(theta)
    a, b, c, d = 2 * ux * ux - 1, 2 * ux * uy, 2 * ux * uy, 2 * uy * uy - 1
    e = cx - (a * cx + c * cy)
    f = cy - (b * cx + d * cy)
    return tuple(0.0 if abs(value) < 1e-12 else float(value) for value in (a, b, c, d, e, f))


def _targets(regions: Sequence[VectorRegion]) -> dict[str, _Target]:
    """Derive public target handles from the region tree; do not persist an index."""
    result: dict[str, _Target] = {}

    def visit(region: VectorRegion, label: str, sources: frozenset[str]) -> None:
        if region.is_leaf:
            if label in result:
                raise ValueError(f"duplicate drawing region id: {label}")
            result[label] = _Target(label, region, sources)
            return
        for child in region.children:
            visit(child, f"{label}-{child.id}", sources)

    for region in regions:
        drawing_id = region.drawing_id
        if type(drawing_id) is not str:
            raise ValueError("drawing root is missing a stable drawing_id")
        visit(region, drawing_id, frozenset(region.source_regions or (drawing_id,)))
    return result


def _replace_leaf(
    regions: Sequence[VectorRegion], target_id: str, transform: Callable[[VectorRegion], VectorRegion]
) -> DrawingRegions:
    """Replace one labeled leaf while retaining the rest of the forest by value."""
    changed = False

    def visit(region: VectorRegion, label: str) -> VectorRegion:
        nonlocal changed
        if region.is_leaf:
            if label == target_id:
                changed = True
                return transform(region)
            return region
        children = tuple(visit(child, f"{label}-{child.id}") for child in region.children)
        return region.with_children(children) if children != region.children else region

    replaced = tuple(visit(region, _root_label(region)) for region in regions)
    if not changed:
        raise KeyError(target_id)
    return replaced


def _group_regions(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Create a semantic merged target while retaining its raster provenance."""
    targets = _targets(regions)
    members = tuple(targets[region_id] for region_id in op["regions"])
    member_ids = {member.id for member in members}
    roots = {_root_label(region): region for region in regions}
    if any(member_id not in roots for member_id in member_ids):
        raise ValueError("group currently requires root drawing regions")
    member_regions = tuple(roots[member.id] for member in members)
    first = member_regions[0]
    assert first.current is not None and first.fill is not None
    raster = np.zeros_like(first.raster, dtype=bool)
    for member in member_regions:
        raster |= member.raster
    merged = VectorRegion.from_shape(
        id=max(region.id for region in _all_regions(regions)) + 1,
        shape=first.current,
        fill=first.fill,
        z=min(member.z for member in member_regions),
        raster=raster,
        drawing_id=op["id"],
        source_regions=tuple(source for member in members for source in member.source_regions),
    )
    return tuple(region for region in regions if _root_label(region) not in member_ids) + (merged,)


def _all_regions(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    all_regions: list[VectorRegion] = []
    for region in regions:
        all_regions.append(region)
        all_regions.extend(region.leaves()) if region.is_branch else None
    return tuple(all_regions)


def _root_label(region: VectorRegion) -> str:
    if type(region.drawing_id) is not str:
        raise ValueError("drawing root is missing a stable drawing_id")
    return region.drawing_id


def _with_fill(region: VectorRegion, fill: Fill) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, fill=fill, footprint=region.footprint)


def _with_z(region: VectorRegion, z: float) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, z=z, footprint=region.footprint)


def _primitive_shape(trace: TraceResult, target: _Target, kind: str) -> Shape:
    regions = [region for region in trace.regions if region.id in target.source_regions]
    xs = [x for region in regions for x in np.nonzero(region.mask)[1]]
    ys = [y for region in regions for y in np.nonzero(region.mask)[0]]
    x, y, w, h = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if kind == "rect":
        return Shape("rect", {"x": x, "y": y, "w": w, "h": h})
    if kind == "ellipse":
        return Shape("ellipse", {"cx": x + w / 2, "cy": y + h / 2, "rx": w / 2, "ry": h / 2})
    if kind == "circle":
        return Shape("circle", {"cx": x + w / 2, "cy": y + h / 2, "r": min(w, h) / 2})
    return Shape("polygon", {"points": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]})


def _command_start(trace: TraceResult, command_id: str) -> tuple[float, float]:
    for region in trace.regions:
        commands = region.trace_path.commands
        for index, command in enumerate(commands):
            if command.id == command_id:
                previous = commands[index - 1]
                return (float(previous.values[-2]), float(previous.values[-1]))
    raise KeyError(command_id)


def _path_shape(trace: TraceResult, target: _Target, geometry: Mapping[str, object]) -> Shape:
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
    if kind == "flat":
        return FlatFill(spec["color"])
    if kind == "linear_gradient":
        return LinearGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    if kind == "radial_gradient":
        return RadialGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    return RasterFill(dict(spec["geometry"]), spec["png_b64"])
