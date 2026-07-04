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
    return None


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


def _replacement(obj: VectorRegion, shape: Shape, other_id: int, candidate: _SeamCandidate) -> VectorRegion:
    return obj.with_current(
        shape,
        footprint=to_polygon(shape),
        diagnostics={
            "seams": {
                "accepted": True,
                "paired_with": int(other_id),
                "selected": "midpoint",
                "axis": candidate.axis,
                "gap_before": abs(candidate.a_value - candidate.b_value),
                "seam": candidate.seam,
            }
        },
    )


def seams_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[Proposal]:
    leaves = [obj for obj in sorted(objects, key=lambda current: (float(current.z), int(current.id))) if obj.is_leaf]
    proposals: list[Proposal] = []
    for index, a in enumerate(leaves):
        if a.current is None:
            continue
        for b in leaves[index + 1:]:
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
            proposals.append(
                Proposal(
                    (int(a.id), int(b.id)),
                    [
                        _replacement(a, a_shape, b.id, candidate),
                        _replacement(b, b_shape, a.id, candidate),
                    ],
                )
            )
    return proposals
