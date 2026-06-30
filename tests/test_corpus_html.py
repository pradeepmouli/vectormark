import numpy as np

from scripts.generate_corpus_html import CorpusEntry, generate_corpus_html
from vectormark import Options


def _tiny_disk():
    img = np.full((48, 48, 3), 255, np.uint8)
    yy, xx = np.ogrid[:48, :48]
    img[((yy - 24) ** 2 + (xx - 24) ** 2) <= 12 * 12] = (200, 30, 30)
    return img


def test_generate_corpus_html_writes_render_svg_and_diagnostics(tmp_path):
    index = generate_corpus_html(
        tmp_path / "corpus",
        [
            CorpusEntry(
                "tiny_disk",
                "optimizer",
                _tiny_disk,
                Options(optimizer=True),
            )
        ],
    )

    html = index.read_text()
    svg = (index.parent / "svg" / "optimizer-tiny_disk.svg").read_text()
    png = index.parent / "input" / "optimizer-tiny_disk.png"

    assert "optimizer / tiny_disk" in html
    assert '<img src="input/optimizer-tiny_disk.png"' in html
    assert '<object data="svg/optimizer-tiny_disk.svg"' in html
    assert "<summary>SVG</summary>" in html
    assert "<summary>Diagnostics</summary>" in html
    assert "&quot;elements&quot;" in html
    assert png.exists()
    assert svg.startswith("<svg ")
