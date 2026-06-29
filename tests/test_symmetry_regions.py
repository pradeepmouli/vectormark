import numpy as np
from vectormark.symmetry import Axis2D, _perimeter, reflection_off_count
from scipy import ndimage as ndi

def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

def test_perimeter_of_disk_is_boundary_band():
    m = _disk(60, 60, 30, 30, 20)
    p = _perimeter(m)
    assert 100 < p < 180   # ~2*pi*r, a one-pixel ring

def test_reflection_off_count_zero_for_symmetric_axis():
    m = _disk(60, 60, 30, 30, 20)           # disk: symmetric about every axis through center
    fg = np.nonzero(m)                       # (rows, cols) == (ys, xs)
    fg_xy = (fg[1], fg[0])                   # _axis_mismatch wants (xs, ys)
    dist = ndi.distance_transform_edt(~m)
    axis = Axis2D(theta=0.0, cx=30.0, cy=30.0)        # horizontal line through center
    assert reflection_off_count(fg_xy, axis, dist) == 0

def test_reflection_off_count_large_for_wrong_axis():
    # an L-shaped (asymmetric) region: reflection lands well off-shape
    m = np.zeros((60, 60), bool); m[10:50, 10:20] = True; m[40:50, 10:45] = True
    fg = np.nonzero(m); fg_xy = (fg[1], fg[0])
    dist = ndi.distance_transform_edt(~m)
    cy, cx = [c.mean() for c in fg[::-1]][::-1]   # centroid (cy, cx)
    axis = Axis2D(theta=np.pi / 2, cx=float(cx), cy=float(cy))  # vertical through centroid
    assert reflection_off_count(fg_xy, axis, dist) > _perimeter(m)
