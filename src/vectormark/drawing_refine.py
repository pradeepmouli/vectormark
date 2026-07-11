"""Apply drawing plans directly to cached :class:`VectorRegion` roots.

The interactive drawing state is the optimizer's native region forest.  SVG and
reports are derived artifacts, never a second editable scene representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .components import decompose_components
from .contour import outer_contour, region_contours, region_corner_radius
from .drawing_plan import DrawingPlan, PlanValidationError, validate_plan
from .drawing_trace import TraceResult
from .emit import render_svg_doc, resolve_fill, resolve_use_shape, shape_to_svg
from .fit import Shape, _fmt, fit_path, recognize_polygon
from .optimizer.vector_region import VectorRegion, to_polygon
from .optimizer.framework import optimize
from .optimizer.passes import clones_pass, primitives_pass, seams_pass, simplify_pass, split_compound_pass, symmetry_pass
from .optimizer.shape_transform import bake_shape_transform
from .pipeline import Options, _segment_image
from .skia_geometry import SkPath, affinity, unary_union
from .surface_merge import merge_surfaces
from .types import Region
from .refine import half_ellipse_cap_fit, rounded_trapezoid_fit


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


def root_regions(
    trace: TraceResult,
    rgb: np.ndarray | None = None,
    *,
    min_region_fraction: float = 0.02,
) -> DrawingRegions:
    """Create retained roots, optionally reducing one-shot surfaces into roots.

    The raw trace deliberately preserves every palette fragment so path-command
    provenance is available on demand.  Its regions are therefore not the
    agent-facing roots.  When pixels are available, create those roots from the
    same segmentation and surface-merge contract as the one-shot pipeline.
    """
    if rgb is None:
        return _raw_root_regions(trace)
    if not 0.0 <= min_region_fraction < 1.0:
        raise ValueError("min_region_fraction must be at least 0 and less than 1")
    if rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("surface merge RGB dimensions must match the trace canvas")

    pipeline_options = Options(
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
        cubic_paths=trace.options.curve_type == "cubic",
        aa_contours=trace.options.trace_level == "subpixel",
        max_colors=trace.options.max_colors,
        min_region_fraction=min_region_fraction,
    )
    _, _, raw_regions = _segment_image(rgb, pipeline_options)
    surfaces: list[tuple[Region, Fill]] = []
    for component in decompose_components(raw_regions, (trace.height, trace.width)):
        surfaces.extend(merge_surfaces([(region, FlatFill(region.color_hex)) for region in component], rgb))

    roots: list[tuple[Region, Fill, tuple[str, ...]]] = []
    for surface, fill in surfaces:
        source_regions = tuple(
            region.id for region in trace.regions if np.any(surface.mask & region.mask)
        )
        if not source_regions:
            continue
        roots.append((surface, fill, source_regions))

    vector_regions: list[VectorRegion] = []
    for index, (surface, fill, source_regions) in enumerate(
        sorted(roots, key=lambda item: (-item[0].area, item[0].label, item[0].color_hex)), start=1
    ):
        shape = _surface_shape(surface, trace)
        if shape is None:
            continue
        vector_regions.append(
            VectorRegion.from_shape(
                id=index,
                shape=shape,
                fill=fill,
                z=index - 1,
                raster=surface.mask,
                source_label=surface.label,
                color_hex=surface.color_hex,
                drawing_id=f"r{index}",
                source_regions=source_regions,
                diagnostics={"surface_merge": {"sources": source_regions, "merged": len(source_regions) > 1}},
            )
        )
    return tuple(vector_regions)


def _raw_root_regions(trace: TraceResult) -> DrawingRegions:
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


def _surface_shape(surface: Region, trace: TraceResult) -> Shape | None:
    contours = [contour for contour in region_contours(surface.mask) if len(contour) >= 3]
    if not contours:
        return None
    paths = [
        str(
            fit_path(
                contour,
                epsilon=trace.options.simplify_tolerance,
                max_error=trace.options.curve_tolerance,
                cubic=trace.options.curve_type == "cubic",
            ).params["d"]
        )
        for contour in contours
    ]
    params: dict[str, object] = {"d": " ".join(paths)}
    if len(paths) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


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
                "diagnostics": _public_diagnostics(target.region.diagnostics),
            }
        )
    return RenderedDrawing(
        render_svg_doc(trace.width, trace.height, body, defs),
        {"targets": tuple(report_targets)},
    )


def drawing_summary(trace: TraceResult, regions: Sequence[VectorRegion]) -> dict[str, object]:
    """Return the compact, merged-root view agents receive after interactive trace."""
    targets = _targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    summaries: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        shape = target.region.current
        geometry: dict[str, object] = {"type": shape.kind}
        if shape.kind == "path":
            geometry["d"] = shape.params["d"]
            if shape.params.get("fill_rule"):
                geometry["fill_rule"] = shape.params["fill_rule"]
        summaries.append(
            {
                "id": target.id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": geometry,
                "fill": _public_fill(target.region.fill),
            }
        )
    return {"width": trace.width, "height": trace.height, "options": asdict(trace.options), "regions": summaries}


def labeled_drawing_svg(trace: TraceResult, regions: Sequence[VectorRegion]) -> str:
    """Render a root-level label map, rather than exposing raw trace fragments."""
    targets = _targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    id_map = {target.region.id: target.id for target in ordered}
    body: list[str] = []
    for target in ordered:
        assert target.region.current is not None
        shape = resolve_use_shape(target.region.current, id_map)
        body.append(f'<g opacity="0.45">{shape_to_svg(shape, "#4C9AFF", f"map-{target.id}")}</g>')
        footprint = target.region.footprint
        if isinstance(footprint, SkPath) and not footprint.is_empty:
            min_x, min_y, max_x, max_y = footprint.bounds
            body.append(
                f'<text x="{_fmt((min_x + max_x) / 2)}" y="{_fmt((min_y + max_y) / 2)}" '
                f'text-anchor="middle" dominant-baseline="middle">{target.id}</text>'
            )
    return render_svg_doc(trace.width, trace.height, body)


def refine(trace: TraceResult, base: Sequence[VectorRegion], plan: DrawingPlan) -> DrawingRegions:
    """Execute *plan* as immutable transformations over native vector regions."""
    regions = tuple(base)
    validate_plan(plan, trace, regions)
    z_order: tuple[str, ...] | None = None

    for index, raw in enumerate(plan.ops):
        op = raw
        name = op["op"]
        epsilon, max_error = _tolerances(trace, plan, op)
        if name == "merge":
            regions = _group_regions(regions, op)
        elif name == "split":
            regions = _run_detection(regions, trace, "split", op["target"], epsilon=epsilon, max_error=max_error)
        elif name in {"detect_primitives", "detect_symmetry", "detect_clones"}:
            regions = _run_detection(regions, trace, name, op.get("target"), epsilon=epsilon, max_error=max_error)
        elif name == "set_symmetry":
            regions = _set_symmetry(regions, op)
        elif name == "clone":
            regions = _clone(regions, op)
        elif name == "set_geometry":
            target = _targets(regions)[op["target"]]
            geometry = op["geometry"]
            try:
                shape = (
                    _path_shape(trace, target, geometry, epsilon=epsilon, max_error=max_error)
                    if geometry["type"] == "path"
                    else _primitive_shape(target, geometry["type"], epsilon=epsilon, max_error=max_error)
                )
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/geometry/type", str(error)) from error
            regions = _replace_leaf(
                regions,
                target.id,
                lambda region: region.with_current(
                    shape,
                    footprint=to_polygon(shape),
                    diagnostics={"geometry": {"explicit": geometry["type"]}},
                ),
            )
            path_ops = geometry.get("ops", ()) if geometry["type"] == "path" else ()
            if any(path_op["op"] == "simplify" for path_op in path_ops):
                regions = _run_detection(regions, trace, "simplify", target.id, epsilon=epsilon, max_error=max_error)
            if any(path_op["op"] == "seams" for path_op in path_ops):
                regions = _run_detection(regions, trace, "seams", target.id, epsilon=epsilon, max_error=max_error)
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
    *,
    epsilon: float,
    max_error: float,
) -> DrawingRegions:
    """Run one existing optimizer pass over the cached region forest."""
    if operation in {"simplify", "seams"}:
        regions = _bake_self_symmetry_uses_for_polish(regions)
    cubic = trace.options.curve_type == "cubic"
    if operation == "detect_primitives":
        def pass_fn(objects, masks):
            return primitives_pass(objects, masks, epsilon=epsilon)
    elif operation == "detect_symmetry":
        def pass_fn(objects, masks):
            return symmetry_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    elif operation == "detect_clones":
        def pass_fn(objects, masks):
            return clones_pass(objects, masks, symbolic=True)
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
            proposals = original_pass(objects, masks)
            if operation != "detect_clones":
                return [proposal for proposal in proposals if selected_region_id in proposal.obj_ids]
            return [
                proposal
                for proposal in proposals
                if selected_region_id in proposal.obj_ids
                or any(
                    replacement.diagnostics.get("clones", {}).get("matched_source") == selected_region_id
                    for replacement in proposal.new_objects
                )
            ]

    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    optimized = optimize(list(regions), masks, [pass_fn])
    return tuple(_restore_root_metadata(region, before.get(region.id)) for region in optimized)


def _bake_self_symmetry_uses_for_polish(regions: Sequence[VectorRegion]) -> DrawingRegions:
    """Bake only self-symmetry mirrors before geometry polishing.

    Clones stay as SVG ``<use>`` relations.  A self-symmetry mirror, however,
    shares a seam with its source half; seam/simplify passes must see both
    concrete paths to avoid preserving or exposing a renderer hairline.
    """
    def bake_root(root: VectorRegion) -> VectorRegion:
        leaves = {leaf.id: leaf for leaf in root.leaves()}

        def visit(region: VectorRegion) -> VectorRegion:
            if region.is_branch:
                children = tuple(visit(child) for child in region.children)
                return region.with_children(children) if children != region.children else region
            assert region.current is not None
            symmetry = region.diagnostics.get("symmetry")
            if (
                region.current.kind != "use"
                or not isinstance(symmetry, Mapping)
                or symmetry.get("mode") != "self_mirror"
            ):
                return region
            source_id = region.current.params.get("href_obj_id")
            source = leaves.get(int(source_id)) if isinstance(source_id, int) else None
            if source is None or source.current is None:
                raise ValueError("self-symmetry mirror is missing its source geometry")
            shape = bake_shape_transform(source.current, tuple(region.current.params["transform"]))
            baked_diagnostics = dict(symmetry)
            baked_diagnostics["baked_for_polish"] = True
            return region.with_current(shape, footprint=region.footprint, diagnostics={"symmetry": baked_diagnostics})

        return visit(root)

    return tuple(bake_root(root) for root in regions)


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
    """Create a semantic target with a SkPath-unioned, seam-free outline."""
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
    footprints = [member.footprint for member in member_regions]
    if not all(isinstance(footprint, SkPath) for footprint in footprints):
        raise ValueError("merge requires polygonal vector-region footprints")
    footprint = unary_union([footprint for footprint in footprints if isinstance(footprint, SkPath)])
    shape = Shape("path", {"d": footprint.to_svg_d()})
    merged = VectorRegion.from_shape(
        id=max(region.id for region in _all_regions(regions)) + 1,
        shape=shape,
        fill=first.fill,
        z=min(member.z for member in member_regions),
        raster=raster,
        footprint=footprint,
        drawing_id=op["id"],
        source_regions=tuple(source for member in members for source in member.source_regions),
        diagnostics={"merge": {"unioned": True, "sources": tuple(member.id for member in members)}},
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


def _primitive_shape(target: _Target, kind: str, *, epsilon: float, max_error: float) -> Shape:
    """Fit one public primitive from the target's retained root mask."""
    mask = target.region.raster
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"cannot fit {kind!r} to an empty drawing region")
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    if kind == "rect":
        return Shape("rect", {"x": x, "y": y, "w": w, "h": h})
    if kind == "rounded_rect":
        radius = region_corner_radius(mask)
        return Shape("rect", {"x": x, "y": y, "w": w, "h": h, "rx": radius, "ry": radius})
    if kind == "ellipse":
        return Shape("ellipse", {"cx": x + w / 2, "cy": y + h / 2, "rx": w / 2, "ry": h / 2})
    if kind == "circle":
        return Shape("circle", {"cx": x + w / 2, "cy": y + h / 2, "r": min(w, h) / 2})
    contour = outer_contour(mask)
    if len(contour) < 3:
        raise ValueError(f"cannot fit {kind!r} without an outer contour")
    axis_x = float((contour[:, 0].min() + contour[:, 0].max()) / 2.0)
    if kind in {"trapezoid", "rounded_trapezoid"}:
        radius = 0.0 if kind == "trapezoid" else region_corner_radius(mask)
        fitted = rounded_trapezoid_fit(contour, axis_x, radius=radius, max_error=max_error)
        if fitted is not None:
            return fitted
        raise ValueError(f"target {target.id!r} is not a {kind}")
    if kind == "cap":
        fitted = half_ellipse_cap_fit(
            contour,
            axis_x,
            corner_radius=region_corner_radius(mask),
            max_error=max_error,
        )
        if fitted is not None:
            return fitted
        raise ValueError(f"target {target.id!r} is not a cap")
    fitted = recognize_polygon(contour, epsilon=epsilon)
    if fitted is not None:
        return fitted
    raise ValueError(f"target {target.id!r} is not a polygon within epsilon={epsilon}")


def _command_start(trace: TraceResult, command_id: str) -> tuple[float, float]:
    for region in trace.regions:
        commands = region.trace_path.commands
        for index, command in enumerate(commands):
            if command.id == command_id:
                previous = commands[index - 1]
                return (float(previous.values[-2]), float(previous.values[-1]))
    raise KeyError(command_id)


def _path_shape(
    trace: TraceResult,
    target: _Target,
    geometry: Mapping[str, object],
    *,
    epsilon: float,
    max_error: float,
) -> Shape:
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
                pieces.append(str(fit_path(points, epsilon=epsilon, max_error=max_error, cubic=kind == "cubic").params["d"]))
        elif op["op"] == "close":
            close = True
    d = " ".join(pieces)
    if close and not d.endswith("Z"):
        d += " Z"
    return Shape("path", {"d": d})


def _tolerances(trace: TraceResult, plan: DrawingPlan, op: Mapping[str, object]) -> tuple[float, float]:
    """Resolve trace defaults, plan defaults, then an operation override."""
    defaults = plan.defaults
    epsilon = float(defaults.get("epsilon", trace.options.simplify_tolerance))
    max_error = float(defaults.get("max_error", trace.options.curve_tolerance))
    return float(op.get("epsilon", epsilon)), float(op.get("max_error", max_error))


def _fill(spec: Mapping[str, object]) -> Fill:
    kind = spec["type"]
    if kind == "flat":
        return FlatFill(spec["color"])
    if kind == "linear_gradient":
        return LinearGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    if kind == "radial_gradient":
        return RadialGradientFill(dict(spec["geometry"]), list(spec["stops"]))
    return RasterFill(dict(spec["geometry"]), spec["png_b64"])


def _public_fill(fill: Fill | None) -> dict[str, object]:
    if isinstance(fill, FlatFill):
        return {"type": "flat", "color": fill.hex}
    if isinstance(fill, LinearGradientFill):
        return {"type": "linear_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RadialGradientFill):
        return {"type": "radial_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RasterFill):
        return {"type": "raster", "geometry": dict(fill.geometry)}
    return {"type": "none"}


def _public_diagnostics(value: object) -> object:
    """Make optimizer diagnostics safe to return in an MCP structured result."""
    if isinstance(value, Mapping):
        return {str(key): _public_diagnostics(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_public_diagnostics(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return repr(value)
