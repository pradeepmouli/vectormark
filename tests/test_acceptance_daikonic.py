import re
from pathlib import Path

import numpy as np
from PIL import Image

from vectormark import Options, idealize
from tests._render import render_svg, ssim, mean_delta_e

FIX = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
ICON_BOTTOM = 392  # the wordmark sits below this row; idealize only the mark


def _icon_array():
    """The Daikonic icon WITHOUT the wordmark. The wordmark is asymmetric, so
    feeding the full image would defeat vertical-symmetry detection (no <use>
    mirror). `idealize` accepts a numpy array directly, so we crop and pass it."""
    src = np.asarray(Image.open(FIX).convert("RGB"), dtype=np.uint8)
    return src[:ICON_BOTTOM]


def test_daikonic_structure_and_symmetry():
    svg = idealize(_icon_array(), options=Options(epsilon=1.8, max_error=1.2))
    # at least the two colour bands recognized as native rect/polygon
    assert svg.count("<rect") + svg.count("<polygon") >= 1
    # bilateral symmetry: the leaves emit a <use> mirror
    assert "<use" in svg
    # unified palette: exactly one navy hex present (AA navies collapsed)
    navies = set(re.findall(r"#0[0-9A-F]2[0-9A-F]3[0-9A-F]", svg.upper()))
    assert len(navies) <= 1
    # the four brand colours survive, nothing extra
    fills = {f.upper() for f in re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg)}
    assert len(fills) == 4
    # a teal is present (green & blue high, red low) — exact hex varies by the
    # palette mode, so match the hue family rather than a literal hex
    def _rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    assert any(g > 110 and b > 110 and r < 120 for r, g, b in map(_rgb, fills))


def test_daikonic_renders_close_to_source():
    icon = _icon_array()
    h, w, _ = icon.shape
    svg = idealize(icon, options=Options(epsilon=1.8, max_error=1.2))
    out = render_svg(svg, w, h)
    score = ssim(icon, out)
    de = mean_delta_e(icon, out)
    assert score >= 0.90, f"SSIM too low: {score:.3f}"
    assert de <= 0.05, f"mean ΔE too high: {de:.3f}"
