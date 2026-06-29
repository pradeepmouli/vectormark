from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..candidate import Fill
from ..emit import shape_to_path_d
from ..fit import Shape

_NUM = r"-?\d+(?:\.\d+)?"


def _sample_subpath(
    tokens: list[tuple[str, str]],
    samples: int,
) -> list[tuple[float, float]]:
    cur = start = None
    pts: list[tuple[float, float]] = []
    for kind, args in tokens:
        values = [float(x) for x in re.findall(_NUM, args)]
        if kind == "M":
            cur = np.array(values[:2])
            start = cur
            pts.append(tuple(cur))
        elif kind == "L":
            cur = np.array(values[:2])
            pts.append(tuple(cur))
        elif kind == "Q":
            assert cur is not None
            c = np.array(values[:2])
            p = np.array(values[2:4])
            for t in np.linspace(0, 1, samples)[1:]:
                pts.append(tuple((1 - t) ** 2 * cur + 2 * (1 - t) * t * c + t**2 * p))
            cur = p
        elif kind == "C":
            assert cur is not None
            c1 = np.array(values[:2])
            c2 = np.array(values[2:4])
            p = np.array(values[4:6])
            for t in np.linspace(0, 1, samples)[1:]:
                pts.append(
                    tuple(
                        (1 - t) ** 3 * cur
                        + 3 * (1 - t) ** 2 * t * c1
                        + 3 * (1 - t) * t**2 * c2
                        + t**3 * p
                    )
                )
            cur = p
        elif kind == "Z":
            cur = start
    return pts


def flatten_points(shape: Shape, *, samples: int = 24) -> list[tuple[float, float]]:
    d = shape_to_path_d(shape)
    tokens = re.findall(rf"([MLQCZ])((?:\s*{_NUM}){{0,6}})", d)
    return _sample_subpath(tokens, samples)


def to_polygon(shape: Shape, *, samples: int = 24) -> Polygon | MultiPolygon:
    d = shape_to_path_d(shape)
    tokens = re.findall(rf"([MLQCZ])((?:\s*{_NUM}){{0,6}})", d)
    subpaths: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for token in tokens:
        if token[0] == "M" and current:
            subpaths.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        subpaths.append(current)

    rings = [_sample_subpath(subpath, samples) for subpath in subpaths]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return Polygon()

    polys = [Polygon(ring) for ring in rings]
    polys = [poly if poly.is_valid else poly.buffer(0) for poly in polys]
    shell = max(polys, key=lambda poly: poly.area)
    holes = [poly for poly in polys if poly is not shell]
    out: Polygon | MultiPolygon = shell
    for hole in holes:
        out = out.difference(hole)
    return out if out.is_valid else out.buffer(0)


@dataclass(frozen=True)
class OptObject:
    id: int
    exact: Shape
    fill: Fill
    z: int
    flat: Polygon | MultiPolygon | object = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.flat is None:
            object.__setattr__(self, "flat", to_polygon(self.exact))

    def with_exact(self, new_shape: Shape) -> "OptObject":
        return OptObject(self.id, new_shape, self.fill, self.z)
