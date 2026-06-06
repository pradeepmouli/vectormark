# SPDX-License-Identifier: MIT
import numpy as np


def test_mean_delta_e_zero_for_identical_and_positive_for_different():
    from vectormark.color import mean_delta_e
    a = np.full((4, 4, 3), 120, np.uint8)
    assert mean_delta_e(a, a) == 0.0
    b = a.copy()
    b[:, :, 0] = 200          # shift red channel
    assert mean_delta_e(a, b) > 0.02


def test_linear_gradient_def_emits_stops_and_coords():
    from vectormark.emit import linear_gradient_def
    d = linear_gradient_def("g0", 10, 20, 110, 20, [(0.0, "#ff0000"), (1.0, "#0000ff")])
    assert d.startswith("<linearGradient") and 'id="g0"' in d
    assert 'gradientUnits="userSpaceOnUse"' in d
    assert 'x1="10"' in d and 'x2="110"' in d
    assert d.count("<stop") == 2
    assert 'offset="0"' in d and 'stop-color="#ff0000"' in d
    assert 'offset="1"' in d and 'stop-color="#0000ff"' in d


def test_radial_gradient_def_emits_center_radius():
    from vectormark.emit import radial_gradient_def
    d = radial_gradient_def("g1", 50, 60, 40, [(0.0, "#ffffff"), (1.0, "#000000")])
    assert d.startswith("<radialGradient") and 'id="g1"' in d
    assert 'cx="50"' in d and 'cy="60"' in d and 'r="40"' in d
    assert d.count("<stop") == 2


def test_render_svg_doc_wraps_defs():
    from vectormark.emit import render_svg_doc
    out = render_svg_doc(100, 100, ['<rect/>'], defs=['<linearGradient id="g0"></linearGradient>'])
    assert "<defs>" in out and "</defs>" in out
    assert out.index("<defs>") < out.index("<rect/>")     # defs before body
    out2 = render_svg_doc(100, 100, ['<rect/>'])
    assert "<defs>" not in out2                            # no defs block when none given
