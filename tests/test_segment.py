import numpy as np
from vectormark.color import extract_palette, quantize
from vectormark.segment import segment, hexstr


def _logo_on_white():
    """48x48 white bg, a navy square and a separate teal square."""
    img = np.full((48, 48, 3), 255, dtype=np.uint8)
    img[6:20, 6:42] = (6, 35, 54)     # navy bar
    img[28:42, 6:42] = (61, 168, 157)  # teal bar
    return img


def test_segment_drops_white_background_and_returns_two_regions():
    img = _logo_on_white()
    q = quantize(img, extract_palette(img))
    regions = segment(q)
    assert len(regions) == 2
    assert {r.color_hex for r in regions} == {"#062336", "#3DA89D"}


def test_hexstr_formats_uppercase():
    assert hexstr((6, 35, 54)) == "#062336"
