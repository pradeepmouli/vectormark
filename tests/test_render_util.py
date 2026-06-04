import numpy as np
from tests._render import render_svg, ssim, mean_delta_e

TEAL = "#3DA89D"
SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="20" height="20"><rect width="20" height="20" fill="{TEAL}"/></svg>'


def test_render_returns_rgb_array():
    img = render_svg(SVG, 20, 20)
    assert img.shape == (20, 20, 3)
    assert img.dtype == np.uint8
    # center pixel ~ teal (61,168,157)
    r, g, b = img[10, 10]
    assert abs(int(r) - 61) < 6 and abs(int(g) - 168) < 6 and abs(int(b) - 157) < 6


def test_identical_images_score_perfectly():
    img = render_svg(SVG, 20, 20)
    assert ssim(img, img) > 0.999
    assert mean_delta_e(img, img) < 0.01
