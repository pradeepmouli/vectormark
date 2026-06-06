"""Smooth-gradient reconstruction end-to-end through idealize. A smooth (non-posterized)
gradient mark becomes one shape + one <linearGradient>/<radialGradient>; flats and multi-blob
marks stay flat. Each positive test first asserts band-grouping finds nothing, so the gradient
is attributable to the smooth-silhouette path, not the band-grouping path."""

import numpy as np

from vectormark import Options, idealize
from vectormark.color import mean_delta_e
from vectormark.gradient import _ramp_groups
from vectormark.pipeline import _segment_image
from tests._render import render_svg


def _smooth_linear_rect(h, w, x0, x1, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    t = ((xx - x0) / (x1 - x0)).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    rect = np.zeros((h, w), bool); rect[int(h * 0.2):int(h * 0.8), x0:x1] = True
    for ch in range(3):
        img[:, :, ch][rect] = (c0[ch] + t * (c1[ch] - c0[ch]))[rect]
    return img.round().astype(np.uint8)


def _smooth_radial_disc(h, w, c, r, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - c[0], yy - c[1])
    t = (dist / r).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    disc = dist <= r
    for ch in range(3):
        img[:, :, ch][disc] = (c0[ch] + t * (c1[ch] - c0[ch]))[disc]
    return img.round().astype(np.uint8)


def test_smooth_linear_rect_via_smooth_path():
    h, w = 160, 240
    # Low-contrast so the 16-colour palette collapses to ≤2 bands (no band-grouping).
    # Original (90,150,230)→(60,110,205) posterised into 3 bands and triggered _ramp_groups.
    img = _smooth_linear_rect(h, w, 40, 200, (85, 145, 225), (70, 125, 210))
    sw, sh, regions = _segment_image(img, Options())
    assert _ramp_groups(regions) == []                  # band-grouping does NOT fire
    svg = idealize(img, options=Options())
    assert svg.count("<linearGradient") == 1            # ...so the smooth path produced it
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.06


def test_smooth_radial_disc_via_smooth_path():
    h, w = 200, 200
    # Low-contrast so the palette collapses to ≤2 bands (no band-grouping).
    # Original (120,190,245)→(70,120,210) posterised into 4 bands and triggered _ramp_groups.
    img = _smooth_radial_disc(h, w, (100, 100), 85, (115, 185, 240), (85, 140, 215))
    sw, sh, regions = _segment_image(img, Options())
    assert _ramp_groups(regions) == []                  # band-grouping does NOT fire
    svg = idealize(img, options=Options())
    assert svg.count("<radialGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.07


def test_smooth_two_blob_stays_flat():
    h, w = 160, 260
    img = np.full((h, w, 3), 255, np.uint8)
    g = np.linspace(0.0, 1.0, 80)
    for x0 in (20, 160):                                 # two disconnected smooth squares
        block = np.empty((80, 80, 3))
        for ch, (a, b) in enumerate(zip((85, 145, 225), (70, 125, 210))):
            block[:, :, ch] = a + g[None, :] * (b - a)
        img[40:120, x0:x0 + 80] = block.round().astype(np.uint8)
    svg = idealize(img, options=Options())
    assert "<linearGradient" not in svg and "<radialGradient" not in svg   # dom 0.5 < 0.85
