import numpy as np
from PIL import Image, ImageDraw

from vectormark.variants import (
    DEFAULT_EPSILONS, DEFAULT_MAX_ERRORS, Variant, generate_variants,
)


def _mark(n=80):
    im = Image.new("RGB", (n, n), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((8, 8, n - 8, n - 8), fill=(30, 100, 220))
    d.rectangle((30, 30, 50, 70), fill=(220, 40, 40))
    return np.asarray(im, dtype=np.uint8)


def test_generate_variants_grid_shape_and_order():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert len(variants) == 4
    # row-major: epsilon outer, max_error inner
    assert [(v.epsilon, v.max_error) for v in variants] == [
        (0.5, 0.5), (0.5, 2.5), (3.0, 0.5), (3.0, 2.5),
    ]
    for v in variants:
        assert isinstance(v, Variant)
        assert v.svg.startswith("<svg ")
        assert v.error is None
        assert v.report.elements >= 1


def test_generate_variants_defaults_are_3x3():
    variants = generate_variants(_mark())
    assert len(variants) == len(DEFAULT_EPSILONS) * len(DEFAULT_MAX_ERRORS) == 9


def test_generate_variants_params_take_effect():
    variants = generate_variants(_mark(), epsilons=(0.3, 6.0), max_errors=(0.5,))
    # a very loose epsilon must change the geometry vs a very tight one
    assert variants[0].svg != variants[1].svg


import json

from vectormark.variants import write_variant_set


def test_write_variant_set_writes_svgs_and_manifest(tmp_path):
    variants = generate_variants(_mark(), epsilons=(0.5, 3.0), max_errors=(1.0,))
    write_variant_set(variants, tmp_path, source="mark.png")

    assert (tmp_path / "variant-e0_5-m1.svg").read_text().startswith("<svg ")
    assert (tmp_path / "variant-e3-m1.svg").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source"] == "mark.png"
    assert manifest["axes"] == {"epsilon": [0.5, 3.0], "max_error": [1.0]}
    assert len(manifest["variants"]) == 2
    first = manifest["variants"][0]
    assert first["epsilon"] == 0.5 and first["max_error"] == 1.0
    assert first["file"] == "variant-e0_5-m1.svg"
    assert first["svg_bytes"] > 0
    assert isinstance(first["strategies"], dict) and first["elements"] >= 1


import io

import pytest

import vectormark.variants as V
from vectormark.score import SvgRendererUnavailable


def _renderer_available():
    try:
        import resvg_py  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _renderer_available(), reason="needs resvg-py")
def test_contact_sheet_renders_grid_png():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    png = V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 0
    img = Image.open(io.BytesIO(bytes(png)))
    assert img.width > 0 and img.height > 0


def test_contact_sheet_none_without_renderer(monkeypatch):
    def boom(*a, **k):
        raise SvgRendererUnavailable("no renderer")
    monkeypatch.setattr(V, "_rasterize", boom)
    eps, mes = (0.5,), (0.5,)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes) is None
