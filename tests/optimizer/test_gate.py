import numpy as np
from shapely.geometry import Point, Polygon

from vectormark.optimizer.gate import BUDGET, coverage_residual, gate_ok


def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_gate_accepts_matching_circle():
    h = w = 120
    true = _disk_mask(h, w, 60, 60, 40)
    circ = Point(60, 60).buffer(40, quad_segs=64)

    assert coverage_residual(circ, true) < 0.05
    assert gate_ok(circ, true)


def test_gate_rejects_wrong_shape():
    h = w = 120
    true = _disk_mask(h, w, 60, 60, 40)
    square = Polygon([(20, 20), (100, 20), (100, 100), (20, 100)])

    assert coverage_residual(square, true) > BUDGET
    assert not gate_ok(square, true)
