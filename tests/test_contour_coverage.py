import numpy as np
from vectormark.contour import region_contours, outer_contour
from vectormark.types import Region


def _disc_mask(r=20, H=80):
    yy, xx = np.ogrid[:H, :H]
    return ((yy - 40) ** 2 + (xx - 40) ** 2) <= r ** 2


def test_coverage_none_is_byte_identical_to_mask():
    m = _disc_mask()
    a = region_contours(m)
    b = region_contours(m, coverage=None)
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca, cb)


def test_coverage_field_is_traced_smoother_than_binary():
    # a smooth coverage field (soft disc) yields a contour closer to the true circle
    H = 80
    yy, xx = np.ogrid[:H, :H]
    d = np.sqrt((yy - 40.0) ** 2 + (xx - 40.0) ** 2)
    cov = np.clip(0.5 + (20.0 - d), 0, 1)          # 0.5 isocontour at radius 20, smooth
    mask = d <= 20
    cs_cov = outer_contour(mask, coverage=cov)
    cs_bin = outer_contour(mask)
    def rms_radius_err(c):
        rr = np.hypot(c[:, 0] - 40.0, c[:, 1] - 40.0)
        return float(np.sqrt(np.mean((rr - 20.0) ** 2)))
    assert rms_radius_err(cs_cov) < rms_radius_err(cs_bin)   # smoother by construction
