import numpy as np
from vectormark.color import srgb_to_oklab, oklab_to_srgb, delta_e


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
