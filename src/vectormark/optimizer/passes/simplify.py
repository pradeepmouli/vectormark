from __future__ import annotations

import math
import re

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ...fit import Shape, _fmt, fit_path
from ..framework import Proposal
from ..gate import BUDGET, rasterize
from ..vector_region import VectorRegion, _parse_subpaths, _ring_area, _sample_subpath

_PATH_COMMAND = re.compile(r"[MLQCAZ]")


def _command_count(d: str) -> int:
    return len(_PATH_COMMAND.findall(d))


def _closed_points(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 4:
        return None
    arr = np.asarray(points, dtype=float)
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    parts: list[str] = []
    for command, values in tokens:
        if values:
            parts.append(f"{command}{' '.join(_fmt(v) for v in values)}")
        else:
            parts.append(command)
    return " ".join(parts)


def _forced_corners(tokens: list[tuple[str, list[float]]]) -> np.ndarray | None:
    if not any(command in {"Q", "C", "A"} for command, _values in tokens):
        return None
    corners: list[tuple[float, float]] = []
    for command, values in tokens:
        if command == "M" and len(values) >= 2:
            corners.append((float(values[0]), float(values[1])))
        elif command == "L" and len(values) >= 2:
            corners.append((float(values[0]), float(values[1])))
    if not corners:
        return None
    return np.asarray(corners, dtype=float)


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    denom = float(np.dot(span, span))
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _curve_is_straight(points: list[np.ndarray], start: np.ndarray, end: np.ndarray, epsilon: float) -> bool:
    return all(_point_line_distance(point, start, end) <= epsilon for point in points)


def _normalized_subpath_d(tokens: list[tuple[str, list[float]]], *, epsilon: float) -> str:
    parts: list[str] = []
    current: np.ndarray | None = None
    start: np.ndarray | None = None
    for command, values in tokens:
        if command == "M":
            current = np.array(values[:2], dtype=float)
            start = current.copy()
            parts.append(f"M{_fmt(current[0])} {_fmt(current[1])}")
        elif command == "L":
            current = np.array(values[:2], dtype=float)
            parts.append(f"L{_fmt(current[0])} {_fmt(current[1])}")
        elif command == "Q" and current is not None:
            control = np.array(values[:2], dtype=float)
            end = np.array(values[2:4], dtype=float)
            if _curve_is_straight([control], current, end, epsilon):
                parts.append(f"L{_fmt(end[0])} {_fmt(end[1])}")
            else:
                parts.append(_subpath_d([(command, values)]))
            current = end
        elif command == "C" and current is not None:
            control1 = np.array(values[:2], dtype=float)
            control2 = np.array(values[2:4], dtype=float)
            end = np.array(values[4:6], dtype=float)
            if _curve_is_straight([control1, control2], current, end, epsilon):
                parts.append(f"L{_fmt(end[0])} {_fmt(end[1])}")
            else:
                parts.append(_subpath_d([(command, values)]))
            current = end
        elif command == "Z":
            parts.append("Z")
            current = start
        else:
            parts.append(_subpath_d([(command, values)]))
    return " ".join(parts)


def _path_subpaths(
    shape: Shape,
    *,
    samples: int = 24,
) -> list[tuple[list[tuple[str, list[float]]], list[tuple[float, float]], float]]:
    subpaths = []
    for tokens in _parse_subpaths(str(shape.params.get("d", ""))):
        points = _sample_subpath(tokens, samples=samples)
        if len(points) < 3:
            continue
        subpaths.append((tokens, points, _ring_area(points)))
    return subpaths


def _quadratic_arc_segment(
    cx: float,
    cy: float,
    radius: float,
    a0: float,
    a1: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    p0 = np.array([cx + radius * math.cos(a0), cy + radius * math.sin(a0)], dtype=float)
    p1 = np.array([cx + radius * math.cos(a1), cy + radius * math.sin(a1)], dtype=float)
    d0 = np.array([-math.sin(a0), math.cos(a0)], dtype=float)
    d1 = np.array([-math.sin(a1), math.cos(a1)], dtype=float)
    matrix = np.column_stack([d0, -d1])
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        control = (p0 + p1) / 2.0
    else:
        t, _ = np.linalg.solve(matrix, p1 - p0)
        control = p0 + float(t) * d0
    return (float(control[0]), float(control[1])), (float(p1[0]), float(p1[1]))


def _append_quadratic_arc(parts: list[str], cx: float, cy: float, radius: float, a0: float, a1: float) -> None:
    mid = (a0 + a1) / 2.0
    for start, end in ((a0, mid), (mid, a1)):
        control, point = _quadratic_arc_segment(cx, cy, radius, start, end)
        parts.append(f"Q{_fmt(control[0])} {_fmt(control[1])} {_fmt(point[0])} {_fmt(point[1])}")


def _side_radius_estimates(points: np.ndarray, x0: float, y0: float, x1: float, y1: float, tol: float) -> list[float]:
    estimates: list[float] = []
    width = x1 - x0
    height = y1 - y0
    sides = (
        (points[points[:, 1] <= y0 + tol][:, 0], width, lambda lo, hi: (lo - x0, x1 - hi)),
        (points[points[:, 1] >= y1 - tol][:, 0], width, lambda lo, hi: (lo - x0, x1 - hi)),
        (points[points[:, 0] <= x0 + tol][:, 1], height, lambda lo, hi: (lo - y0, y1 - hi)),
        (points[points[:, 0] >= x1 - tol][:, 1], height, lambda lo, hi: (lo - y0, y1 - hi)),
    )
    for values, span, fn in sides:
        if len(values) < 2:
            continue
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < span * 0.20:
            continue
        estimates.extend(r for r in fn(lo, hi) if r > tol)
    return estimates


def _rounded_rect_radius(points: np.ndarray, poly: Polygon, epsilon: float) -> tuple[float, tuple[float, float, float, float]] | None:
    x0, y0, x1, y1 = (float(v) for v in poly.bounds)
    width = x1 - x0
    height = y1 - y0
    min_dim = min(width, height)
    if width < 4.0 or height < 4.0:
        return None

    bbox_area = width * height
    if bbox_area <= 0.0:
        return None
    fill_ratio = float(poly.area) / bbox_area
    if not 0.65 <= fill_ratio < 0.995:
        return None

    tol = max(1.0, epsilon * 1.5, min_dim * 0.01)
    estimates = _side_radius_estimates(points, x0, y0, x1, y1, tol)
    deficit = max(0.0, bbox_area - float(poly.area))
    area_radius = math.sqrt(deficit / (4.0 - math.pi)) if deficit > 0.0 else 0.0
    if area_radius > tol:
        estimates.append(area_radius)
    if not estimates:
        return None

    radius = max(float(np.median(estimates)), area_radius)
    if radius < max(2.0, min_dim * 0.025) or radius > min_dim * 0.48:
        return None
    return radius, (x0, y0, x1, y1)


def _rounded_rect_path(points: list[tuple[float, float]], *, epsilon: float) -> str | None:
    contour = _closed_points(points)
    if contour is None:
        return None
    ring = contour[:-1]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not isinstance(poly, Polygon) or poly.is_empty:
        return None

    fit = _rounded_rect_radius(ring, poly, epsilon)
    if fit is None:
        return None
    radius, (x0, y0, x1, y1) = fit

    parts = [f"M{_fmt(x0 + radius)} {_fmt(y0)}", f"L{_fmt(x1 - radius)} {_fmt(y0)}"]
    _append_quadratic_arc(parts, x1 - radius, y0 + radius, radius, -math.pi / 2.0, 0.0)
    parts.append(f"L{_fmt(x1)} {_fmt(y1 - radius)}")
    _append_quadratic_arc(parts, x1 - radius, y1 - radius, radius, 0.0, math.pi / 2.0)
    parts.append(f"L{_fmt(x0 + radius)} {_fmt(y1)}")
    _append_quadratic_arc(parts, x0 + radius, y1 - radius, radius, math.pi / 2.0, math.pi)
    parts.append(f"L{_fmt(x0)} {_fmt(y0 + radius)}")
    _append_quadratic_arc(parts, x0 + radius, y0 + radius, radius, math.pi, math.pi * 1.5)
    parts.append("Z")
    return " ".join(parts)


def _candidate_shape(
    shape: Shape,
    parts: list[str],
    *,
    preserve_fill_rule: bool = True,
) -> Shape:
    params = {"d": " ".join(parts)}
    if preserve_fill_rule and shape.params.get("fill_rule"):
        params["fill_rule"] = shape.params["fill_rule"]
    return Shape("path", params)


def _ordered_subpath_indices(subpaths) -> list[int]:
    if not subpaths:
        return []
    outer_index = max(range(len(subpaths)), key=lambda i: subpaths[i][2])
    return [outer_index, *(i for i in range(len(subpaths)) if i != outer_index)]


def _simplified_subpath_d(
    tokens: list[tuple[str, list[float]]],
    points: list[tuple[float, float]],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> str:
    original = _subpath_d(tokens)
    candidates = [_normalized_subpath_d(tokens, epsilon=epsilon)]
    rounded = _rounded_rect_path(points, epsilon=epsilon)
    if rounded is not None:
        candidates.append(rounded)

    contour = _closed_points(points)
    if contour is not None:
        candidates.append(
            str(
                fit_path(
                    contour,
                    epsilon=epsilon,
                    max_error=max_error,
                    cubic=cubic,
                    forced_corners=_forced_corners(tokens),
                ).params["d"]
            )
        )

    best = min(candidates, key=_path_complexity_d)
    return best if _path_complexity_d(best) < _path_complexity_d(original) else original


def _candidate_parts(
    subpaths,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    preserve_subpaths: bool,
) -> list[str]:
    indices = _ordered_subpath_indices(subpaths)
    if not preserve_subpaths:
        indices = indices[:1]
    parts: list[str] = []
    for index in indices:
        tokens, points, _area = subpaths[index]
        parts.append(
            _simplified_subpath_d(
                tokens,
                points,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
            )
        )
    return parts


def _path_complexity(shape: Shape) -> tuple[int, int] | None:
    if shape.kind != "path":
        return None
    d = str(shape.params.get("d", ""))
    return _path_complexity_d(d)


def _path_complexity_d(d: str) -> tuple[int, int]:
    return _command_count(d), len(d.encode())


def _is_improvement(current: Shape, candidate: Shape, *, original: Shape | None = None) -> bool:
    candidate_complexity = _path_complexity(candidate)
    current_complexity = _path_complexity(current)
    if candidate_complexity is None or current_complexity is None:
        return False
    if candidate == current:
        return False
    if not (
        candidate_complexity[0] < current_complexity[0]
        and candidate_complexity[1] < current_complexity[1]
    ):
        return False
    original_complexity = _path_complexity(original) if original is not None else None
    if original_complexity is None:
        return True
    return (
        candidate_complexity[0] <= original_complexity[0]
        and candidate_complexity[1] <= original_complexity[1]
    )


def _simplified_path_shape(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    samples: int,
    preserve_subpaths: bool = True,
    cubic: bool = False,
    original: Shape | None = None,
) -> Shape | None:
    if shape.kind != "path":
        return None

    d = str(shape.params.get("d", ""))
    if not d:
        return None

    subpaths = _path_subpaths(shape, samples=samples)
    if not subpaths:
        return None
    parts = _candidate_parts(
        subpaths,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        preserve_subpaths=preserve_subpaths,
    )
    if not parts:
        return None
    candidate = _candidate_shape(shape, parts, preserve_fill_rule=preserve_subpaths)
    if not _is_improvement(shape, candidate, original=original):
        return None
    return candidate


def _simplifiable_polygon(flat: object) -> bool:
    return isinstance(flat, Polygon) and not flat.is_empty


def _covering_later_ids(
    obj: VectorRegion,
    candidate: Shape,
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[int] | None:
    if obj.id not in masks:
        return None
    candidate_obj = obj.with_current(candidate)
    candidate_mask = rasterize(candidate_obj.footprint, masks[obj.id].shape)
    added = candidate_mask & ~np.asarray(masks[obj.id], dtype=bool)
    added_count = int(added.sum())
    if added_count == 0:
        return []

    covered = np.zeros_like(added, dtype=bool)
    cover_ids: list[int] = []
    for other in sorted(objects, key=lambda item: (float(item.z), int(item.id))):
        if other.id == obj.id or other.id not in masks or other.z <= obj.z:
            continue
        other_mask = np.asarray(masks[other.id], dtype=bool)
        if (other_mask & added).any():
            cover_ids.append(int(other.id))
            covered |= other_mask

    uncovered = int((added & ~covered).sum())
    base_area = max(1, int(np.asarray(masks[obj.id], dtype=bool).sum()))
    if uncovered / base_area > BUDGET:
        return None
    return cover_ids


def _referenced_source_ids(objects: list[VectorRegion]) -> set[int]:
    referenced: set[int] = set()
    for obj in objects:
        if not obj.is_leaf or obj.current is None or obj.current.kind != "use":
            continue
        href_obj_id = obj.current.params.get("href_obj_id")
        if href_obj_id is None:
            continue
        referenced.add(int(href_obj_id))
    return referenced


def simplify_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    samples: int = 16,
    cubic: bool = False,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    referenced_source_ids = _referenced_source_ids(objects)
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        if int(obj.id) in referenced_source_ids:
            continue
        if isinstance(obj.footprint, MultiPolygon) or not _simplifiable_polygon(obj.footprint):
            continue
        if obj.current.kind == "path" and obj.current.params.get("fill_rule"):
            solid = _simplified_path_shape(
                obj.current,
                epsilon=epsilon,
                max_error=max_error,
                samples=samples,
                preserve_subpaths=False,
                cubic=cubic,
                original=obj.original,
            )
            if solid is not None:
                cover_ids = _covering_later_ids(obj, solid, objects, masks)
                if cover_ids is not None:
                    by_id = {int(current.id): current for current in objects}
                    proposal_ids = tuple([int(obj.id), *cover_ids])
                    new_objects = [obj.with_current(solid), *(by_id[obj_id] for obj_id in cover_ids)]
                    proposals.append(Proposal(proposal_ids, new_objects))
                    continue
        simplified = _simplified_path_shape(
            obj.current,
            epsilon=epsilon,
            max_error=max_error,
            samples=samples,
            cubic=cubic,
            original=obj.original,
        )
        if simplified is None:
            continue
        proposals.append(Proposal((obj.id,), [obj.with_current(simplified)]))

    return proposals
