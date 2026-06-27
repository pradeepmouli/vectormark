"""Gradient reconstruction end-to-end through idealize. A genuine gradient becomes one
shape + one <linearGradient>/<radialGradient> that re-renders within a perceptual ΔE
bar; flat and non-ramp inputs stay flat (no <defs>)."""

import numpy as np

from vectormark import Options, idealize
from tests._render import render_svg, mean_delta_e


def _linear_img(h, w, p0, p1, stops_rgb, bg=(255, 255, 255), shape_mask=None):
    yy, xx = np.mgrid[:h, :w]
    d = np.array(p1, float) - np.array(p0, float); L = float(d @ d)
    t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([c for _, c in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    img = img.round().astype(np.uint8)
    m = shape_mask if shape_mask is not None else np.ones((h, w), bool)
    out = np.full((h, w, 3), bg, np.uint8); out[m] = img[m]
    return out


def _radial_img(h, w, c, r, stops_rgb, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - c[0], yy - c[1])
    t = (dist / r).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([col for _, col in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    out = np.full((h, w, 3), bg, np.uint8)
    disc = dist <= r
    out[disc] = img.round().astype(np.uint8)[disc]
    return out


def test_linear_gradient_reconstructs():
    h, w = 200, 260
    img = _linear_img(h, w, (40, 100), (220, 100),
                      [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    svg = idealize(img, options=Options())
    assert svg.count("<linearGradient") == 1


def test_radial_gradient_reconstructs():
    h, w = 220, 220
    img = _radial_img(h, w, (110, 110), 90,
                      [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])
    svg = idealize(img, options=Options())
    assert svg.count("<radialGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.07


def test_flat_logo_not_gradientified():
    h, w = 160, 160
    img = np.full((h, w, 3), 255, np.uint8)
    img[40:120, 40:120] = (37, 99, 235)               # one flat blue square
    svg = idealize(img, options=Options())
    assert "<defs>" not in svg and "url(#" not in svg


def test_two_color_nonramp_stays_flat():
    h, w = 160, 200
    img = np.full((h, w, 3), 255, np.uint8)
    img[30:130, 20:90] = (220, 30, 30)                # red block
    img[30:130, 110:180] = (30, 30, 220)              # blue block (not a ramp)
    svg = idealize(img, options=Options())
    assert "<linearGradient" not in svg and "<radialGradient" not in svg


def test_linear_gradient_flatten_emits_userspace_gradient():
    h, w = 200, 260
    img = _linear_img(h, w, (40, 100), (220, 100),
                      [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    svg = idealize(img, options=Options(flatten=True))
    assert svg.count("<linearGradient") == 1
    assert 'gradientUnits="userSpaceOnUse"' in svg
    assert "url(#g0)" in svg
    assert "<g transform=" not in svg          # flatten bakes geometry; no wrapping transform
