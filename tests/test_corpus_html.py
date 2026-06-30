import numpy as np
from PIL import Image

import scripts.generate_corpus_html as corpus_html
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


def test_default_entries_use_real_logo_corpus_and_include_vbird(tmp_path, monkeypatch):
    corpus = tmp_path / "scratch" / "real-logos"
    corpus.mkdir(parents=True)
    Image.fromarray(_tiny_disk()).save(corpus / "icloud.png")
    Image.fromarray(_tiny_disk()).save(corpus / "vbird.png")
    Image.fromarray(_tiny_disk()).save(corpus / "cmp_icloud.png")

    monkeypatch.setattr(corpus_html, "REPO_ROOT", tmp_path)

    entries = corpus_html.default_entries()

    assert [entry.name for entry in entries].count("icloud") == 2
    assert [entry.name for entry in entries].count("vbird") == 2
    assert "cmp_icloud" not in {entry.name for entry in entries}
    assert {entry.mode for entry in entries if entry.name == "vbird"} == {"current", "optimizer"}
