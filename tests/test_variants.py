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
