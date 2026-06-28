import numpy as np
from vectormark.types import Region
from vectormark.pipeline import attach_coverage_field
from vectormark.segment import hexstr


def _region(mask, rgb):
    return Region(label=1, mask=mask, color_hex=hexstr(rgb))


def test_flat_region_gets_coverage():
    H, W = 60, 80
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[15:45, 20:60] = (20, 40, 200)
    mask = np.zeros((H, W), bool); mask[15:45, 20:60] = True
    r = _region(mask, (20, 40, 200))
    attach_coverage_field([r], rgb, max_colors=16)
    assert r.coverage is not None
    assert r.coverage[mask].mean() > 0.8


def test_gradient_region_is_hole_guarded_to_mask():
    # A block whose interior sweeps between two palette-present colors -> the K-way field
    # would punch holes -> guard must leave coverage=None (mask contour).
    H, W = 60, 120
    rgb = np.full((H, W, 3), 255, np.uint8)
    # left half ~colorA, right half ~colorB, both inside ONE region (a merged gradient surface)
    rgb[15:45, 10:60] = (10, 60, 200)
    rgb[15:45, 60:110] = (10, 120, 240)   # a second, similar blue present in the palette
    mask = np.zeros((H, W), bool); mask[15:45, 10:110] = True
    r = _region(mask, (10, 60, 200))
    attach_coverage_field([r], rgb, max_colors=16)
    assert r.coverage is None, "gradient surface must keep the mask (hole-guard)"
