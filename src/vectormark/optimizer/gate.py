from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from skimage.draw import polygon as draw_polygon

BUDGET = 0.02


def _fill_ring(coords: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    if len(coords) < 3:
        return mask

    rr, cc = draw_polygon(coords[:, 1], coords[:, 0], shape=shape_hw)
    mask[rr, cc] = True
    return mask


def _rasterize_polygon(poly: Polygon, shape_hw: tuple[int, int]) -> np.ndarray:
    if poly.is_empty:
        return np.zeros(shape_hw, dtype=bool)

    mask = _fill_ring(np.asarray(poly.exterior.coords), shape_hw)
    for interior in poly.interiors:
        hole_mask = _fill_ring(np.asarray(interior.coords), shape_hw)
        mask &= ~hole_mask
    return mask


def rasterize(geom, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    if geom.is_empty:
        return mask
    if isinstance(geom, Polygon):
        return _rasterize_polygon(geom, shape_hw)
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            mask |= _rasterize_polygon(poly, shape_hw)
        return mask
    raise TypeError(f"unsupported geometry type: {type(geom)!r}")


def coverage_residual(geom, true_mask: np.ndarray) -> float:
    truth = np.asarray(true_mask, dtype=bool)
    pred = rasterize(geom, truth.shape)
    diff = np.logical_xor(pred, truth)
    denom = int(truth.sum())
    if denom == 0:
        return 0.0 if not diff.any() else 1.0
    return float(diff.sum() / denom)


def gate_ok(geom, true_mask, *, budget: float = BUDGET) -> bool:
    return coverage_residual(geom, true_mask) <= budget
