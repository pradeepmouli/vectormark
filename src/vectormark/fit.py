"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import CircleModel, EllipseModel


@dataclass
class Shape:
    kind: str                 # "circle" | "ellipse" | "rect" | "polygon" | "path"
    params: dict
    closed: bool = True


def _max_residual(model, pts: np.ndarray) -> float:
    return float(np.abs(model.residuals(pts)).max())


def recognize_primitive(contour: np.ndarray, *, epsilon: float) -> Shape | None:
    """Return a native-primitive Shape if `contour` matches one within ε, else None."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 8:
        return None
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 1:
        return None

    # circle
    cm = CircleModel()
    if cm.estimate(pts) and _max_residual(cm, pts) <= epsilon:
        xc, yc, r = cm.params
        return Shape("circle", {"cx": xc, "cy": yc, "r": r})

    # ellipse (axis-aligned check: snap small thetas to 0 for symmetric output)
    em = EllipseModel()
    if em.estimate(pts):
        xc, yc, a, b, theta = em.params
        if _max_residual(em, pts) <= epsilon and (abs(theta) < 0.08 or abs(abs(theta) - np.pi) < 0.08):
            return Shape("ellipse", {"cx": xc, "cy": yc, "rx": a, "ry": b})

    # axis-aligned rectangle: bbox fill ratio near 1 and rotated-rect ~ axis-aligned
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    rot = poly.minimum_rotated_rectangle
    rx, ry = rot.exterior.xy
    edge_angles = np.arctan2(np.diff(ry), np.diff(rx))
    axis_aligned = np.all(np.minimum(np.abs(edge_angles % (np.pi / 2)),
                                     np.pi / 2 - np.abs(edge_angles % (np.pi / 2))) < 0.06)
    if bbox_area > 0 and poly.area / bbox_area > 0.96 and axis_aligned:
        return Shape("rect", {"x": minx, "y": miny, "w": maxx - minx, "h": maxy - miny})

    return None
