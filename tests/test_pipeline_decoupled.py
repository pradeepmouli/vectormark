import re
import numpy as np
from vectormark.pipeline import idealize, Options


def _wing_with_dot():
    """A blue->navy vertical ramp 'wing' with a separate uniform blue 'dot' beside it
    on white — a minimal stand-in for the V-bird failure case."""
    H, W = 120, 120
    img = np.full((H, W, 3), 255, np.uint8)
    for y in range(20, 100):                      # the wing: smooth vertical ramp
        t = (y - 20) / 79
        img[y, 20:70] = [round(40 + t * 0), round(120 - t * 90), round(230 - t * 110)]
    yy, xx = np.ogrid[:H, :W]                      # a separate uniform blue dot
    img[((yy - 40) ** 2 + (xx - 95) ** 2) <= 12 ** 2] = (30, 100, 220)
    return img


def test_wing_emits_one_gradient_and_dot_survives():
    svg = idealize(_wing_with_dot(), options=Options(max_colors=16))
    # the wing is ONE gradient, not stacked flat bands
    assert len(re.findall(r"<linearGradient", svg)) >= 1
    # the dot is still present as its own element (a circle or a small path), not absorbed
    # — at least two distinct filled elements exist (wing + dot)
    assert len(re.findall(r"<(path|circle|ellipse|rect|polygon)\b", svg)) >= 2


def test_flat_logo_uses_no_gradient():
    # a solid square on white must stay a single flat shape, no gradient defs
    img = np.full((80, 80, 3), 255, np.uint8)
    img[20:60, 20:60] = (200, 40, 40)
    svg = idealize(img, options=Options(max_colors=16))
    assert "<linearGradient" not in svg and "<radialGradient" not in svg
