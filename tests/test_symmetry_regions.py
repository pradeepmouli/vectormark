import numpy as np
from vectormark.symmetry import Axis2D, _perimeter, reflection_off_count
from vectormark.symmetry import region_is_self_symmetric, regions_mirror_pair, K_BAND
from vectormark.types import Region
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


def _region(mask, label=1):
    return Region(label=label, mask=mask, color_hex="#000000")


def test_disk_is_self_symmetric_about_any_central_axis():
    m = _disk(80, 80, 40, 40, 25)
    for theta in (0.0, np.pi / 4, np.pi / 2):
        assert region_is_self_symmetric(_region(m), Axis2D(theta, 40.0, 40.0))


def test_asymmetric_region_is_not_self_symmetric():
    m = np.zeros((80, 80), bool); m[20:60, 20:30] = True; m[50:60, 20:55] = True  # L
    fg = np.nonzero(m); cy, cx = float(fg[0].mean()), float(fg[1].mean())
    assert not region_is_self_symmetric(_region(m), Axis2D(np.pi / 2, cx, cy))


def test_mirror_pair_detected_and_size_guarded():
    left = np.zeros((80, 80), bool); left[30:50, 15:25] = True
    right = np.zeros((80, 80), bool); right[30:50, 55:65] = True   # mirror of left about x=40
    axis = Axis2D(np.pi / 2, 40.0, 40.0)                            # vertical through x=40
    assert regions_mirror_pair(_region(left, 1), _region(right, 2), axis)
    big = np.zeros((80, 80), bool); big[20:60, 50:70] = True        # bigger -> not a pair
    assert not regions_mirror_pair(_region(left, 1), _region(big, 3), axis)
