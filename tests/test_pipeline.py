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


def test_solid_color_image_does_not_crash(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((24, 24, 3), (40, 40, 40), np.uint8)).save(tmp_path / "solid.png")
    svg = idealize(str(tmp_path / "solid.png"))
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")

def test_gradient_image_does_not_crash():
    grad = np.zeros((40, 40, 3), np.uint8)
    for x in range(40):
        grad[:, x] = (x * 6, 100, 255 - x * 6)
    svg = idealize(grad)
    assert svg.startswith("<svg")
