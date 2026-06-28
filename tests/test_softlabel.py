import numpy as np
from vectormark.softlabel import alpha_unmix


def test_pure_colors_give_0_and_1():
    ca = np.array([20.0, 40.0, 200.0]); cb = np.array([255.0, 255.0, 255.0])
    assert abs(alpha_unmix(ca, ca, cb) - 1.0) < 1e-9
    assert abs(alpha_unmix(cb, ca, cb) - 0.0) < 1e-9


def test_midpoint_gives_half():
    ca = np.array([0.0, 0.0, 0.0]); cb = np.array([255.0, 255.0, 255.0])
    mid = (ca + cb) / 2
    assert abs(alpha_unmix(mid, ca, cb) - 0.5) < 1e-9


def test_vectorized_and_clipped():
    ca = np.array([0.0, 0.0, 0.0]); cb = np.array([100.0, 0.0, 0.0])
    V = np.array([[-50.0, 0, 0], [50.0, 0, 0], [150.0, 0, 0]])  # below, mid, above
    a = alpha_unmix(V, ca, cb)
    assert a.shape == (3,)
    assert a[0] == 1.0 and abs(a[1] - 0.5) < 1e-9 and a[2] == 0.0  # clipped to [0,1]
