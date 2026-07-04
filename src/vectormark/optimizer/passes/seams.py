from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...fit import Shape, _fmt
from ..framework import Proposal
from ..vector_region import VectorRegion, _parse_subpaths, to_polygon


_SEAM_TOL = 1.25


@dataclass(frozen=True)
class _SeamCandidate:
    axis: str
    a_side: str
    b_side: str
    a_value: float
    b_value: float
    seam: float
    span_min: float
    span_max: float


@dataclass(frozen=True)
class _LeafRef:
    owner: VectorRegion
    leaf: VectorRegion


def _shape_bounds(obj: VectorRegion) -> tuple[float, float, float, float] | None:
    if obj.footprint is None or getattr(obj.footprint, "is_empty", False):
        return None
    minx, miny, maxx, maxy = obj.footprint.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _candidate_for_pair(a: VectorRegion, b: VectorRegion, *, tol: float) -> _SeamCandidate | None:
    if a.current is None or b.current is None:
        return None
    if a.current.kind == "use" or b.current.kind == "use":
        return None
    try:
        if float(a.footprint.distance(b.footprint)) > tol:
            return None
    except Exception:
        return None

    a_bounds = _shape_bounds(a)
    b_bounds = _shape_bounds(b)
    if a_bounds is None or b_bounds is None:
        return None
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds

    y_span = _overlap(ay0, ay1, by0, by1)
    if y_span is not None:
        if 0.0 < bx0 - ax1 <= tol:
            seam = (ax1 + bx0) / 2.0
            return _SeamCandidate("x", "right", "left", ax1, bx0, seam, *y_span)
        if 0.0 < ax0 - bx1 <= tol:
            seam = (bx1 + ax0) / 2.0
            return _SeamCandidate("x", "left", "right", ax0, bx1, seam, *y_span)

    x_span = _overlap(ax0, ax1, bx0, bx1)
    if x_span is not None:
        if 0.0 < by0 - ay1 <= tol:
            seam = (ay1 + by0) / 2.0
            return _SeamCandidate("y", "bottom", "top", ay1, by0, seam, *x_span)
        if 0.0 < ay0 - by1 <= tol:
            seam = (by1 + ay0) / 2.0
            return _SeamCandidate("y", "top", "bottom", ay0, by1, seam, *x_span)
    if a.current.kind != "path" or b.current.kind != "path":
        return None
    return _SeamCandidate("vertices", "", "", 0.0, float(a.footprint.distance(b.footprint)), 0.0, 0.0, 0.0)


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    parts: list[str] = []
    for command, values in tokens:
        if values:
            parts.append(f"{command}{' '.join(_fmt(value) for value in values)}")
        else:
            parts.append(command)
    return " ".join(parts)


def _rewrite_path_side(
    shape: Shape,
    *,
    axis: str,
    side_value: float,
    seam: float,
    span_min: float,
    span_max: float,
    tol: float,
) -> Shape | None:
    if shape.kind != "path":
        return None
    changed = False
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    rewritten: list[list[tuple[str, list[float]]]] = []
    for subpath in subpaths:
        out_subpath: list[tuple[str, list[float]]] = []
        for command, values in subpath:
            new_values = list(values)
            coord_indexes: tuple[tuple[int, int], ...]
            if command in {"M", "L"}:
                coord_indexes = ((0, 1),)
            elif command == "Q":
                coord_indexes = ((0, 1), (2, 3))
            elif command == "C":
                coord_indexes = ((0, 1), (2, 3), (4, 5))
            elif command == "A":
                coord_indexes = ((5, 6),)
            else:
                coord_indexes = ()
            for x_idx, y_idx in coord_indexes:
                x = float(new_values[x_idx])
                y = float(new_values[y_idx])
                if axis == "x":
                    if abs(x - side_value) <= tol and span_min - tol <= y <= span_max + tol:
                        new_values[x_idx] = seam
                        changed = True
                elif abs(y - side_value) <= tol and span_min - tol <= x <= span_max + tol:
                    new_values[y_idx] = seam
                    changed = True
            out_subpath.append((command, new_values))
        rewritten.append(out_subpath)
    if not changed:
        return None

    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in rewritten)
    return Shape("path", params)


def _rewrite_rect_side(shape: Shape, *, side: str, seam: float) -> Shape | None:
    if shape.kind != "rect":
        return None
    params = dict(shape.params)
    width_key = "w" if "w" in params else "width"
    height_key = "h" if "h" in params else "height"
    x = float(params["x"])
    y = float(params["y"])
    w = float(params[width_key])
    h = float(params[height_key])
    if side == "left":
        right = x + w
        w = right - seam
        x = seam
    elif side == "right":
        w = seam - x
    elif side == "top":
        bottom = y + h
        h = bottom - seam
        y = seam
    elif side == "bottom":
        h = seam - y
    else:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    params.update({"x": x, "y": y, width_key: w, height_key: h})
    return Shape("rect", params)


def _rewrite_ellipse_side(shape: Shape, *, side: str, seam: float) -> Shape | None:
    if shape.kind not in {"circle", "ellipse"}:
        return None
    params = dict(shape.params)
    cx = float(params["cx"])
    cy = float(params["cy"])
    if shape.kind == "circle":
        r = float(params["r"])
        if side == "left":
            right = cx + r
            r = (right - seam) / 2.0
            cx = (right + seam) / 2.0
        elif side == "right":
            left = cx - r
            r = (seam - left) / 2.0
            cx = (left + seam) / 2.0
        elif side == "top":
            bottom = cy + r
            r = (bottom - seam) / 2.0
            cy = (bottom + seam) / 2.0
        elif side == "bottom":
            top = cy - r
            r = (seam - top) / 2.0
            cy = (top + seam) / 2.0
        else:
            return None
        if r <= 0.0:
            return None
        params.update({"cx": cx, "cy": cy, "r": r})
        return Shape("circle", params)

    rx = float(params["rx"])
    ry = float(params["ry"])
    if side == "left":
        right = cx + rx
        rx = (right - seam) / 2.0
        cx = (right + seam) / 2.0
    elif side == "right":
        left = cx - rx
        rx = (seam - left) / 2.0
        cx = (left + seam) / 2.0
    elif side == "top":
        bottom = cy + ry
        ry = (bottom - seam) / 2.0
        cy = (bottom + seam) / 2.0
    elif side == "bottom":
        top = cy - ry
        ry = (seam - top) / 2.0
        cy = (top + seam) / 2.0
    else:
        return None
    if rx <= 0.0 or ry <= 0.0:
        return None
    params.update({"cx": cx, "cy": cy, "rx": rx, "ry": ry})
    return Shape("ellipse", params)


def _rewrite_shape_side(
    shape: Shape,
    *,
    side: str,
    candidate: _SeamCandidate,
    side_value: float,
    tol: float,
) -> Shape | None:
    if shape.kind == "use":
        return None
    if candidate.axis == "vertices":
        return shape
    path = _rewrite_path_side(
        shape,
        axis=candidate.axis,
        side_value=side_value,
        seam=candidate.seam,
        span_min=candidate.span_min,
        span_max=candidate.span_max,
        tol=tol,
    )
    if path is not None:
        return path
    primitive = _rewrite_rect_side(shape, side=side, seam=candidate.seam)
    if primitive is not None:
        return primitive
    return _rewrite_ellipse_side(shape, side=side, seam=candidate.seam)


def _coordinate_indexes(command: str) -> tuple[tuple[int, int], ...]:
    if command in {"M", "L"}:
        return ((0, 1),)
    if command == "Q":
        return ((0, 1), (2, 3))
    if command == "C":
        return ((0, 1), (2, 3), (4, 5))
    if command == "A":
        return ((5, 6),)
    return ()


def _path_coordinate_refs(tokens: list[list[tuple[str, list[float]]]]):
    refs = []
    for subpath_index, subpath in enumerate(tokens):
        for token_index, (command, values) in enumerate(subpath):
            for x_idx, y_idx in _coordinate_indexes(command):
                refs.append((subpath_index, token_index, x_idx, y_idx, float(values[x_idx]), float(values[y_idx])))
    return refs


def _shape_from_tokens(shape: Shape, tokens: list[list[tuple[str, list[float]]]]) -> Shape:
    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in tokens)
    return Shape("path", params)


def _true_up_path_vertices(a_shape: Shape, b_shape: Shape, *, tol: float) -> tuple[Shape, Shape, bool]:
    if a_shape.kind != "path" or b_shape.kind != "path":
        return a_shape, b_shape, False

    a_tokens = _parse_subpaths(str(a_shape.params.get("d", "")))
    b_tokens = _parse_subpaths(str(b_shape.params.get("d", "")))
    a_refs = _path_coordinate_refs(a_tokens)
    b_refs = _path_coordinate_refs(b_tokens)
    pairs: list[tuple[float, tuple, tuple]] = []
    for a_ref in a_refs:
        for b_ref in b_refs:
            dx = a_ref[4] - b_ref[4]
            dy = a_ref[5] - b_ref[5]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if 0.0 < dist <= tol:
                pairs.append((dist, a_ref, b_ref))

    changed = False
    used_a: set[tuple[int, int, int, int]] = set()
    used_b: set[tuple[int, int, int, int]] = set()
    for _dist, a_ref, b_ref in sorted(pairs, key=lambda item: item[0]):
        a_key = tuple(a_ref[:4])
        b_key = tuple(b_ref[:4])
        if a_key in used_a or b_key in used_b:
            continue
        midpoint_x = (a_ref[4] + b_ref[4]) / 2.0
        midpoint_y = (a_ref[5] + b_ref[5]) / 2.0
        a_tokens[a_ref[0]][a_ref[1]][1][a_ref[2]] = midpoint_x
        a_tokens[a_ref[0]][a_ref[1]][1][a_ref[3]] = midpoint_y
        b_tokens[b_ref[0]][b_ref[1]][1][b_ref[2]] = midpoint_x
        b_tokens[b_ref[0]][b_ref[1]][1][b_ref[3]] = midpoint_y
        used_a.add(a_key)
        used_b.add(b_key)
        changed = True

    if not changed:
        return a_shape, b_shape, False
    return _shape_from_tokens(a_shape, a_tokens), _shape_from_tokens(b_shape, b_tokens), True


def _replacement(
    obj: VectorRegion,
    shape: Shape,
    other_id: int,
    candidate: _SeamCandidate,
    *,
    selected: str,
) -> VectorRegion:
    return obj.with_current(
        shape,
        footprint=to_polygon(shape),
        diagnostics={
            "seams": {
                "accepted": True,
                "paired_with": int(other_id),
                "selected": selected,
                "axis": candidate.axis,
                "gap_before": abs(candidate.a_value - candidate.b_value),
                "seam": candidate.seam,
            }
        },
    )


def _leaf_refs(objects: list[VectorRegion]) -> list[_LeafRef]:
    refs: list[_LeafRef] = []

    def visit(owner: VectorRegion, current: VectorRegion) -> None:
        if current.is_leaf:
            refs.append(_LeafRef(owner=owner, leaf=current))
            return
        for child in current.children:
            visit(owner, child)

    for obj in sorted(objects, key=lambda current: (float(current.z), int(current.id))):
        visit(obj, obj)
    return refs


def _replace_leaf(root: VectorRegion, leaf_id: int, replacement: VectorRegion) -> VectorRegion:
    if root.is_leaf:
        return replacement if int(root.id) == int(leaf_id) else root
    children = [
        _replace_leaf(child, leaf_id, replacement)
        for child in root.children
    ]
    return root.with_children(children)


def seams_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    refs = _leaf_refs(objects)
    proposals: list[Proposal] = []
    for index, a_ref in enumerate(refs):
        a = a_ref.leaf
        if a.current is None:
            continue
        for b_ref in refs[index + 1:]:
            b = b_ref.leaf
            if int(a_ref.owner.id) == int(b_ref.owner.id):
                continue
            if b.current is None:
                continue
            candidate = _candidate_for_pair(a, b, tol=_SEAM_TOL)
            if candidate is None:
                continue
            a_shape = _rewrite_shape_side(
                a.current,
                side=candidate.a_side,
                candidate=candidate,
                side_value=candidate.a_value,
                tol=_SEAM_TOL,
            )
            b_shape = _rewrite_shape_side(
                b.current,
                side=candidate.b_side,
                candidate=candidate,
                side_value=candidate.b_value,
                tol=_SEAM_TOL,
            )
            if a_shape is None or b_shape is None:
                continue
            a_shape, b_shape, vertices_changed = _true_up_path_vertices(a_shape, b_shape, tol=_SEAM_TOL)
            selected = "vertex_midpoint" if vertices_changed or candidate.axis == "vertices" else "midpoint"
            a_replacement = _replacement(a, a_shape, b.id, candidate, selected=selected)
            b_replacement = _replacement(b, b_shape, a.id, candidate, selected=selected)
            owner_replacements: dict[int, VectorRegion] = {
                int(a_ref.owner.id): _replace_leaf(a_ref.owner, a.id, a_replacement),
                int(b_ref.owner.id): _replace_leaf(b_ref.owner, b.id, b_replacement),
            }
            proposals.append(
                Proposal(
                    tuple(sorted(owner_replacements)),
                    [owner_replacements[obj_id] for obj_id in sorted(owner_replacements)],
                )
            )
    return proposals
