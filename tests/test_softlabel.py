import numpy as np
from scipy import ndimage
from vectormark.softlabel import alpha_unmix, soft_label_field, region_coverage
from vectormark.contour import outer_contour


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


def test_region_coverage_boundary_at_half():
    img, pal = _two_color_ramp(H=40, W=60)
    L = soft_label_field(img, pal)
    mask_a = np.zeros((40, 60), bool); mask_a[:, :30] = True
    cov = region_coverage(L, 0, mask_a)
    assert np.all(cov[:, 0] > 0.9) and np.all(cov[:, -1] < 0.1)   # 1 inside A, 0 outside


def test_shared_seam_is_point_identical():
    # THE SPINE: region A (color 0) and region B (color 1) share the seam; the sub-arcs
    # along the seam must be identical (φ_B = −φ_A) → no gap, no overlap.
    img, pal = _two_color_ramp(H=40, W=60)
    L = soft_label_field(img, pal)
    mask_a = np.zeros((40, 60), bool); mask_a[:, :30] = True
    mask_b = np.zeros((40, 60), bool); mask_b[:, 30:] = True
    cov_a = region_coverage(L, 0, mask_a)
    cov_b = region_coverage(L, 1, mask_b)
    # along the shared seam column band, cov_b == 1 - cov_a (φ_B = −φ_A exactly)
    seam = slice(28, 32)
    assert np.allclose(cov_b[:, seam], 1.0 - cov_a[:, seam], atol=1e-9)


def test_solid_region_with_palette_twin_stays_interior():
    # Two near-identical dark blues in the palette + a solid block of the first one,
    # spatially far from any pixel of the twin. The block's interior must stay one-hot
    # (coverage ~1), NOT be misread as an antialiasing band.
    import numpy as np
    from vectormark.softlabel import soft_label_field, region_coverage
    H, W = 40, 80
    img = np.full((H, W, 3), 255.0)                 # white bg
    img[10:30, 5:25] = (1, 70, 165)                 # shifted toward twin: nearest=row1, but d1≈1.5×d0 → old d0<0.5*d1 heuristic sees "band"
    img[10:30, 55:75] = (1, 61, 151)               # solid #013D97 block (twin, far away)
    palette = np.array([(255, 255, 255), (1, 75, 172), (1, 61, 151)], float)
    L = soft_label_field(img, palette)
    # interior of block A (away from its own edges) is one-hot to label 1
    interior_A = L[15:25, 10:20, :]
    assert interior_A[..., 1].min() > 0.99, "solid twin-color block interior must stay one-hot"
    # and its region coverage is ~1 on the mask interior (was ~0.5 before the fix)
    # Interior-only mask (3 px inset from block edges, safely outside the 2-px AA band)
    maskA = np.zeros((H, W), bool); maskA[13:27, 8:22] = True
    cov = region_coverage(L, 1, maskA)
    assert cov[maskA].mean() > 0.9
