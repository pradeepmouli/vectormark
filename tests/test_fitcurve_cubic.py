import numpy as np
from vectormark._fitcurve import fit_cubic_beziers, cubic_inflects

def _arc(n=40, deg=90, r=200):
    th = np.linspace(0, np.deg2rad(deg), n)
    return np.c_[r*np.cos(th), r*np.sin(th)] + [60, 40]

def _scurve(n=60):
    x = np.linspace(0, 360, n); y = 90*np.sin(np.deg2rad(x))
    return np.c_[x+40, y+200]

def test_arc_fits_in_one_cubic_low_error():
    cs = fit_cubic_beziers(_arc(), max_error=2.0)
    assert len(cs) == 1
    assert not cubic_inflects(cs[0])

def test_every_emitted_cubic_is_inflection_free():
    for data in (_arc(), _scurve(), _arc(deg=170)):
        for c in fit_cubic_beziers(data, max_error=1.5):
            assert not cubic_inflects(c), "emitted a cubic with an inflection point"

def test_scurve_is_split_not_inflected():
    cs = fit_cubic_beziers(_scurve(), max_error=1.5)
    assert len(cs) >= 2     # the S is split into convex pieces

def test_endpoints_are_pinned_and_shared():
    data = _scurve(); cs = fit_cubic_beziers(data, max_error=1.5)
    assert np.allclose(cs[0][0], data[0]) and np.allclose(cs[-1][3], data[-1])
    for a, b in zip(cs, cs[1:]):
        assert np.allclose(a[3], b[0])     # contiguous

def test_deterministic():
    d = _scurve()
    a = fit_cubic_beziers(d, 1.5); b = fit_cubic_beziers(d, 1.5)
    assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
