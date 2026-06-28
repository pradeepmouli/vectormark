import numpy as np
from vectormark.softlabel import alpha_unmix, soft_label_field


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


def _two_color_ramp(H=40, W=60):
    # left half color A, right half color B, with a 2px antialiased ramp at x=W/2
    A = np.array([20.0, 40.0, 200.0]); B = np.array([255.0, 255.0, 255.0])
    img = np.empty((H, W, 3))
    for x in range(W):
        t = np.clip((x - (W / 2 - 1)) / 2.0, 0, 1)   # 0 left of seam, 1 right, ramp across 2px
        img[:, x] = (1 - t) * A + t * B
    return img, np.array([A, B], np.uint8)


def test_partition_of_unity():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    assert L.shape == (40, 60, 2)
    assert np.allclose(L.sum(axis=2), 1.0, atol=1e-6)        # memberships sum to 1


def test_interior_is_one_hot():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    # far-left column is pure A -> L[...,0] == 1 (interior anchoring)
    assert np.allclose(L[:, 0, 0], 1.0) and np.allclose(L[:, 0, 1], 0.0)
    assert np.allclose(L[:, -1, 0], 0.0) and np.allclose(L[:, -1, 1], 1.0)


def test_seam_band_crosses_half():
    img, pal = _two_color_ramp()
    L = soft_label_field(img, pal)
    # along a row, A's membership decreases monotonically across the seam and passes 0.5
    row = L[20, :, 0]
    assert row[0] > 0.9 and row[-1] < 0.1
    assert np.any(np.abs(row - 0.5) < 0.1)                   # a 0.5 crossing exists
