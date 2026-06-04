import numpy as np
from vectormark.color import srgb_to_oklab, oklab_to_srgb, delta_e
from vectormark.color import extract_palette, quantize


def test_oklab_roundtrip():
    rgb = np.array([[0.024, 0.137, 0.212], [0.99, 0.55, 0.15], [1, 1, 1]])
    back = oklab_to_srgb(srgb_to_oklab(rgb))
    assert np.allclose(rgb, back, atol=1e-4)


def test_white_maps_to_L_one():
    lab = srgb_to_oklab(np.array([[1.0, 1.0, 1.0]]))
    assert abs(lab[0, 0] - 1.0) < 1e-3
    assert abs(lab[0, 1]) < 1e-3 and abs(lab[0, 2]) < 1e-3


def test_delta_e_is_symmetric_and_zero_on_equal():
    a = np.array([0.5, 0.1, -0.05])
    b = np.array([0.4, 0.0, 0.02])
    assert delta_e(a, a) == 0.0
    assert abs(delta_e(a, b) - delta_e(b, a)) < 1e-12


def _band_image():
    """64x64: navy top, teal bottom, with a realistic 1px anti-alias gradient
    row between — each column is a distinct navy->teal blend, so no single blend
    colour is frequent enough to survive palette extraction (mirrors real AA,
    where edge blends spread across many low-count colours)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:31] = (6, 35, 54)       # navy
    navy = np.array([6.0, 35.0, 54.0])
    teal = np.array([61.0, 168.0, 157.0])
    for x in range(64):
        t = x / 63.0
        img[31, x] = np.round(navy * (1 - t) + teal * t).astype(np.uint8)
    img[32:] = (61, 168, 157)    # teal
    return img


def test_extract_palette_finds_two_true_colors_not_the_blend():
    pal = extract_palette(_band_image(), max_colors=8)
    assert len(pal) == 2
    hexes = {tuple(c) for c in pal}
    assert (6, 35, 54) in hexes and (61, 168, 157) in hexes


def test_quantize_collapses_blend_row():
    img = _band_image()
    pal = extract_palette(img, max_colors=8)
    q = quantize(img, pal)
    assert set(map(tuple, np.unique(q.reshape(-1, 3), axis=0))) <= {(6, 35, 54), (61, 168, 157)}


def test_extract_palette_never_empty_on_gradient():
    grad = np.zeros((40, 40, 3), np.uint8)
    for x in range(40):
        grad[:, x] = (x * 6, 100, 255 - x * 6)
    pal = extract_palette(grad)
    assert len(pal) >= 1
