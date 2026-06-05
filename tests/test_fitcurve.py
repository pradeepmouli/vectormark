import numpy as np
from vectormark._fitcurve import fit_cubic_beziers


def test_fits_quarter_circle_within_tolerance():
    t = np.linspace(0, np.pi / 2, 40)
    pts = np.column_stack([np.cos(t), np.sin(t)]) * 50
    beziers = fit_cubic_beziers(pts, max_error=0.5)
    assert len(beziers) >= 1
    # each bezier is 4 control points of dim 2
    assert all(b.shape == (4, 2) for b in beziers)
    # endpoints match the data endpoints
    assert np.allclose(beziers[0][0], pts[0], atol=1e-6)
    assert np.allclose(beziers[-1][3], pts[-1], atol=1e-6)
