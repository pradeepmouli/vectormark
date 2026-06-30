from __future__ import annotations

import re

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ...fit import Shape, fit_path
from ..framework import Proposal
from ..optobject import OptObject, flatten_points

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


def _simplified_path_shape(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    samples: int,
) -> Shape | None:
    if shape.kind != "path":
        return None

    d = str(shape.params.get("d", ""))
    if not d:
        return None

    points = flatten_points(shape, samples=samples)
    contour = _closed_points(points)
    if contour is None:
        return None

    simplified = fit_path(contour, epsilon=epsilon, max_error=max_error)
    if simplified is None:
        return None
    if _command_count(str(simplified.params["d"])) >= _command_count(d):
        return None
    if simplified == shape:
        return None
    return simplified


def _single_shell(flat: object) -> bool:
    return isinstance(flat, Polygon) and not flat.is_empty and not flat.interiors


def simplify_pass(
    objects: list[OptObject],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    samples: int = 16,
) -> list[Proposal]:
    del masks

    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if isinstance(obj.flat, MultiPolygon) or not _single_shell(obj.flat):
            continue
        simplified = _simplified_path_shape(
            obj.exact,
            epsilon=epsilon,
            max_error=max_error,
            samples=samples,
        )
        if simplified is None:
            continue
        proposals.append(Proposal((obj.id,), [obj.with_exact(simplified)]))

    return proposals
