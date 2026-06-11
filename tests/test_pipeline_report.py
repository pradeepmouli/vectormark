import numpy as np
from PIL import Image, ImageDraw

from vectormark.pipeline import IdealizeReport, idealize


def _disc(n=64):
    im = Image.new("RGB", (n, n), "white")
    ImageDraw.Draw(im).ellipse((10, 10, n - 10, n - 10), fill=(30, 100, 220))
    return np.asarray(im, dtype=np.uint8)


def test_idealize_default_returns_bare_string():
    out = idealize(_disc())
    assert isinstance(out, str) and out.startswith("<svg ")


def test_idealize_report_returns_strategy_counts():
    svg, report = idealize(_disc(), report=True)
    assert isinstance(svg, str) and svg.startswith("<svg ")
    assert isinstance(report, IdealizeReport)
    # a clean disc is one region recognised as a single primitive (circle)
    assert report.strategies == {"primitive": 1}
    assert report.elements == 1
    assert report.gradients == 0


def test_report_empty_for_blank_image():
    blank = np.full((40, 40, 3), 255, dtype=np.uint8)
    svg, report = idealize(blank, report=True)
    assert report.elements == 0 and dict(report.strategies) == {}
