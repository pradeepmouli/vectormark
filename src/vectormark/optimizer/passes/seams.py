from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._fitcurve import cbezier, cubic_inflects, fit_quadratic_beziers
from ...fit import Shape, _curve_controls_are_straight, _fmt
from ..framework import Proposal
from ..shape_transform import bake_shape_transform
from ..vector_region import VectorRegion, _parse_subpaths, to_polygon
from .simplify import _simplified_path_shape


_SEAM_TOL = 2.0
_JUNCTION_TOL = 6.5


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


@dataclass(frozen=True)
class _LeafUpdate:
    shape: Shape
    paired_with: int
    candidate: _SeamCandidate
    selected: str


@dataclass(frozen=True)
class _CoordinateRef:
    key: tuple[int, int]
    subpath_index: int
    token_index: int
    x_index: int
    y_index: int
    x: float
    y: float


def _shape_bounds(obj: VectorRegion) -> tuple[float, float, float, float] | None:
    if obj.footprint is None or getattr(obj.footprint, "is_empty", False):
        return None
    minx, miny, maxx, maxy = obj.footprint.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _regions_close(a: VectorRegion, b: VectorRegion, *, tol: float) -> bool:
    try:
        return float(a.footprint.distance(b.footprint)) <= tol
    except Exception:
        return False


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
    if not _regions_close(a, b, tol=tol):
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
                coord_indexes = ((2, 3),)
            elif command == "C":
                coord_indexes = ((4, 5),)
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
        return ((2, 3),)
    if command == "C":
        return ((4, 5),)
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


def _path_endpoint_refs(
    key: tuple[int, int],
    tokens: list[list[tuple[str, list[float]]]],
) -> list[_CoordinateRef]:
    refs: list[_CoordinateRef] = []
    for subpath_index, subpath in enumerate(tokens):
        for token_index, (command, values) in enumerate(subpath):
            for x_idx, y_idx in _coordinate_indexes(command):
                refs.append(
                    _CoordinateRef(
                        key,
                        subpath_index,
                        token_index,
                        x_idx,
                        y_idx,
                        float(values[x_idx]),
                        float(values[y_idx]),
                    )
                )
    return refs


def _shape_from_tokens(shape: Shape, tokens: list[list[tuple[str, list[float]]]]) -> Shape:
    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in tokens)
    return Shape("path", params)


def _cleanup_linelets(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    simplified = _simplified_path_shape(
        shape,
        epsilon=epsilon,
        max_error=max_error,
        samples=16,
        cubic=cubic,
        linelet_only=True,
    )
    return simplified or shape


def _cleanup_inflecting_cubics(shape: Shape, *, max_error: float, line_epsilon: float) -> Shape:
    if shape.kind != "path":
        return shape
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    if not subpaths:
        return shape
    changed = False
    rewritten: list[list[tuple[str, list[float]]]] = []
    for subpath in subpaths:
        current: np.ndarray | None = None
        start: np.ndarray | None = None
        out_subpath: list[tuple[str, list[float]]] = []
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
                start = current.copy()
                out_subpath.append((command, list(values)))
                continue
            if command == "Z":
                current = start
                out_subpath.append((command, list(values)))
                continue
            if command == "C" and current is not None:
                ctrl = np.array(
                    [
                        current,
                        values[:2],
                        values[2:4],
                        values[4:6],
                    ],
                    dtype=float,
                )
                if cubic_inflects(ctrl):
                    samples = np.asarray([cbezier(ctrl, t) for t in np.linspace(0.0, 1.0, 9)], dtype=float)
                    for bezier in fit_quadratic_beziers(samples, max_error):
                        if _curve_controls_are_straight(bezier[0], [bezier[1]], bezier[2], line_epsilon):
                            out_subpath.append(("L", [float(bezier[2][0]), float(bezier[2][1])]))
                        else:
                            out_subpath.append(
                                (
                                    "Q",
                                    [
                                        float(bezier[1][0]),
                                        float(bezier[1][1]),
                                        float(bezier[2][0]),
                                        float(bezier[2][1]),
                                    ],
                                )
                            )
                    current = ctrl[3]
                    changed = True
                    continue
            out_subpath.append((command, list(values)))
            if command == "L":
                current = np.array(values[:2], dtype=float)
            elif command == "Q":
                current = np.array(values[2:4], dtype=float)
            elif command == "C":
                current = np.array(values[4:6], dtype=float)
            elif command == "A":
                current = np.array(values[5:7], dtype=float)
        rewritten.append(out_subpath)
    if not changed:
        return shape
    return _shape_from_tokens(shape, rewritten)


def _cleanup_mutated_path(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    cleaned = _cleanup_linelets(shape, epsilon=epsilon, max_error=max_error, cubic=cubic)
    return _cleanup_inflecting_cubics(cleaned, max_error=max_error, line_epsilon=epsilon)


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


def _leaf_pairs_between(
    a_owner: VectorRegion,
    a_region: VectorRegion,
    b_owner: VectorRegion,
    b_region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if not _regions_close(a_region, b_region, tol=tol):
        return []
    if a_region.is_leaf and b_region.is_leaf:
        return [(_LeafRef(owner=a_owner, leaf=a_region), _LeafRef(owner=b_owner, leaf=b_region))]

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    if a_region.is_branch and b_region.is_branch:
        for a_child in a_region.children:
            for b_child in b_region.children:
                pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_child, tol=tol))
        return pairs

    if a_region.is_branch:
        for a_child in a_region.children:
            pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_region, tol=tol))
        return pairs

    for b_child in b_region.children:
        pairs.extend(_leaf_pairs_between(a_owner, a_region, b_owner, b_child, tol=tol))
    return pairs


def _leaf_pairs_within(
    owner: VectorRegion,
    region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if region.is_leaf:
        return []

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    children = list(region.children)
    for index, a_child in enumerate(children):
        for b_child in children[index + 1:]:
            pairs.extend(_leaf_pairs_between(owner, a_child, owner, b_child, tol=tol))
    for child in children:
        pairs.extend(_leaf_pairs_within(owner, child, tol=tol))
    return pairs


def _replace_leaf(root: VectorRegion, leaf_id: int, replacement: VectorRegion) -> VectorRegion:
    if root.is_leaf:
        return replacement if int(root.id) == int(leaf_id) else root
    children = [
        _replace_leaf(child, leaf_id, replacement)
        for child in root.children
    ]
    return root.with_children(children)


def _replace_leaves(root: VectorRegion, replacements_by_leaf_id: dict[int, VectorRegion]) -> VectorRegion:
    if root.is_leaf:
        return replacements_by_leaf_id.get(int(root.id), root)
    children = [
        _replace_leaves(child, replacements_by_leaf_id)
        for child in root.children
    ]
    return root.with_children(children)


def _leaf_key(ref: _LeafRef) -> tuple[int, int]:
    return int(ref.owner.id), int(ref.leaf.id)


def _apply_leaf_pair(
    a_ref: _LeafRef,
    b_ref: _LeafRef,
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_leaf_maps: dict[int, dict[int, VectorRegion]],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> tuple[int, int] | None:
    a = a_ref.leaf
    if a.current is None:
        return None
    b = b_ref.leaf
    if b.current is None:
        return None

    def current_shape(ref: _LeafRef) -> Shape | None:
        key = _leaf_key(ref)
        if key in updates_by_leaf:
            return updates_by_leaf[key].shape
        leaf = ref.leaf
        if leaf.current is None:
            return None
        if leaf.current.kind != "use":
            return leaf.current
        source_id = int(leaf.current.params.get("href_obj_id", -1))
        source = owner_leaf_maps.get(int(ref.owner.id), {}).get(source_id)
        if source is None or source.current is None:
            return None
        source_key = (int(ref.owner.id), source_id)
        source_shape = updates_by_leaf[source_key].shape if source_key in updates_by_leaf else source.current
        return bake_shape_transform(source_shape, leaf.current.params["transform"])

    a_current = current_shape(a_ref)
    b_current = current_shape(b_ref)
    if a_current is None or b_current is None:
        return None
    a_for_candidate = a.with_current(a_current, footprint=a.footprint)
    b_for_candidate = b.with_current(b_current, footprint=b.footprint)
    candidate = _candidate_for_pair(a_for_candidate, b_for_candidate, tol=_SEAM_TOL)
    if candidate is None:
        return None
    a_key = _leaf_key(a_ref)
    b_key = _leaf_key(b_ref)
    a_shape = _rewrite_shape_side(
        a_current,
        side=candidate.a_side,
        candidate=candidate,
        side_value=candidate.a_value,
        tol=_SEAM_TOL,
    )
    b_shape = _rewrite_shape_side(
        b_current,
        side=candidate.b_side,
        candidate=candidate,
        side_value=candidate.b_value,
        tol=_SEAM_TOL,
    )
    if a_shape is None or b_shape is None:
        return None
    a_shape, b_shape, vertices_changed = _true_up_path_vertices(a_shape, b_shape, tol=_SEAM_TOL)
    cleaned_a = _cleanup_mutated_path(a_shape, epsilon=epsilon, max_error=max_error, cubic=cubic)
    cleaned_b = _cleanup_mutated_path(b_shape, epsilon=epsilon, max_error=max_error, cubic=cubic)
    if cleaned_a != a_shape or cleaned_b != b_shape:
        a_shape, b_shape = cleaned_a, cleaned_b
        a_shape, b_shape, cleanup_vertices_changed = _true_up_path_vertices(a_shape, b_shape, tol=_SEAM_TOL)
        vertices_changed = vertices_changed or cleanup_vertices_changed
    selected = "vertex_midpoint" if vertices_changed or candidate.axis == "vertices" else "midpoint"
    updates_by_leaf[a_key] = _LeafUpdate(a_shape, int(b.id), candidate, selected)
    updates_by_leaf[b_key] = _LeafUpdate(b_shape, int(a.id), candidate, selected)
    return int(a_ref.owner.id), int(b_ref.owner.id)


def _owner_components(edges: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, set[int]] = {}
    for value in parent:
        groups.setdefault(find(value), set()).add(value)
    return [groups[key] for key in sorted(groups)]


def _cluster_leaf_vertices(
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_ids: set[int],
    *,
    tol: float,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> None:
    keys = [
        key for key, update in updates_by_leaf.items()
        if key[0] in owner_ids and update.shape.kind == "path"
    ]
    if len(keys) < 2:
        return

    tokens_by_key = {
        key: _parse_subpaths(str(updates_by_leaf[key].shape.params.get("d", "")))
        for key in keys
    }
    changed_keys: set[tuple[int, int]] = set()

    def apply_clusters(*, link_tol: float, min_keys: int, allow_same_key_links: bool) -> None:
        refs: list[_CoordinateRef] = []
        for key, tokens in tokens_by_key.items():
            refs.extend(_path_endpoint_refs(key, tokens))
        if len(refs) < 2:
            return

        parent = list(range(len(refs)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a_ref in enumerate(refs):
            for j in range(i + 1, len(refs)):
                b_ref = refs[j]
                if not allow_same_key_links and a_ref.key == b_ref.key:
                    continue
                dx = a_ref.x - b_ref.x
                dy = a_ref.y - b_ref.y
                if 0.0 < (dx * dx + dy * dy) ** 0.5 <= link_tol:
                    union(i, j)

        groups: dict[int, list[_CoordinateRef]] = {}
        for index, ref in enumerate(refs):
            groups.setdefault(find(index), []).append(ref)

        for group in groups.values():
            if len({ref.key for ref in group}) < min_keys:
                continue
            x = sum(ref.x for ref in group) / len(group)
            y = sum(ref.y for ref in group) / len(group)
            for ref in group:
                tokens = tokens_by_key[ref.key]
                values = tokens[ref.subpath_index][ref.token_index][1]
                values[ref.x_index] = x
                values[ref.y_index] = y
                changed_keys.add(ref.key)

    apply_clusters(link_tol=tol, min_keys=2, allow_same_key_links=False)
    apply_clusters(link_tol=_JUNCTION_TOL, min_keys=3, allow_same_key_links=True)

    for key in changed_keys:
        update = updates_by_leaf[key]
        shape = _shape_from_tokens(update.shape, tokens_by_key[key])
        updates_by_leaf[key] = _LeafUpdate(
            _cleanup_mutated_path(shape, epsilon=epsilon, max_error=max_error, cubic=cubic),
            update.paired_with,
            update.candidate,
            "vertex_cluster",
        )


def _replacement_for_update(ref: _LeafRef, update: _LeafUpdate) -> VectorRegion:
    return _replacement(
        ref.leaf,
        update.shape,
        update.paired_with,
        update.candidate,
        selected=update.selected,
    )


def seams_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.0,
    max_error: float = 1.0,
    cubic: bool = False,
) -> list[Proposal]:
    owners = sorted(objects, key=lambda current: (float(current.z), int(current.id)))
    owner_by_id = {int(owner.id): owner for owner in owners}
    owner_leaf_maps = {
        int(owner.id): {int(leaf.id): leaf for leaf in owner.leaves()}
        for owner in owners
    }
    leaf_ref_by_key: dict[tuple[int, int], _LeafRef] = {}
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate] = {}
    owner_edges: list[tuple[int, int]] = []

    def record_pair(a_ref: _LeafRef, b_ref: _LeafRef) -> None:
        leaf_ref_by_key.setdefault(_leaf_key(a_ref), a_ref)
        leaf_ref_by_key.setdefault(_leaf_key(b_ref), b_ref)
        edge = _apply_leaf_pair(
            a_ref,
            b_ref,
            updates_by_leaf,
            owner_leaf_maps,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
        if edge is not None:
            owner_edges.append(edge)

    for owner in owners:
        for a_ref, b_ref in _leaf_pairs_within(owner, owner, tol=_SEAM_TOL):
            record_pair(a_ref, b_ref)

    for owner_index, a_owner in enumerate(owners):
        for b_owner in owners[owner_index + 1:]:
            leaf_pairs = _leaf_pairs_between(a_owner, a_owner, b_owner, b_owner, tol=_SEAM_TOL)
            for a_ref, b_ref in leaf_pairs:
                record_pair(a_ref, b_ref)

    proposals: list[Proposal] = []
    for owner_ids in _owner_components(owner_edges):
        _cluster_leaf_vertices(
            updates_by_leaf,
            owner_ids,
            tol=_SEAM_TOL,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
        replacements: list[VectorRegion] = []
        for owner_id in sorted(owner_ids):
            owner = owner_by_id[owner_id]
            leaf_replacements: dict[int, VectorRegion] = {}
            for key, update in updates_by_leaf.items():
                if key[0] != owner_id:
                    continue
                leaf_replacements[key[1]] = _replacement_for_update(leaf_ref_by_key[key], update)
            if not leaf_replacements:
                continue
            replacements.append(_replace_leaves(owner, leaf_replacements))
        if replacements:
            proposals.append(Proposal(tuple(sorted(owner_ids)), replacements))
    return proposals
