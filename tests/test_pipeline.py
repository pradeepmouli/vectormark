import numpy as np
from PIL import Image
from vectormark.pipeline import idealize


def _two_band_logo(path):
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)      # navy rect
    img[34:52, 20:60] = (61, 168, 157)  # teal rect
    Image.fromarray(img).save(path)


def test_idealize_emits_two_rects(tmp_path):
    p = tmp_path / "logo.png"
    _two_band_logo(p)
    svg = idealize(str(p))
    assert svg.count("<rect") == 2
    assert "#062336" in svg and "#3DA89D" in svg
    assert 'viewBox="0 0 80 60"' in svg
