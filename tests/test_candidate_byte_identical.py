"""Byte-identical regression net for the Candidate/Fill refactor.

Each case's golden SVG is captured from the pre-refactor code (Task 1) and
committed. The refactor (Tasks 2-3) must keep idealize output == golden, exactly.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectormark import Options, idealize
from tests._render import paint, disk
from tests.test_acceptance_annulus import _ring
from tests.test_acceptance_occlusion import _synthetic_mastercard
from tests.test_acceptance_gradient import _linear_img, _radial_img
from tests.test_acceptance_smooth_gradient import _smooth_linear_rect, _rotate_img

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
DAIKONIC = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"


def _daikonic_icon():
    return np.asarray(Image.open(DAIKONIC).convert("RGB"), dtype=np.uint8)[:392]


def _ring_disk():
    h, w = 160, 200
    return paint([(_ring(70, 70, 45, 25, h, w), (37, 99, 235)),
                  (disk(135, 70, 38, h, w), (220, 30, 30))], h, w)


def _linear_grad():
    return _linear_img(200, 260, (40, 100), (220, 100),
                       [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])


def _radial_grad():
    return _radial_img(220, 220, (110, 110), 90,
                       [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])


def _rectified_grad():
    base = _smooth_linear_rect(160, 240, 40, 200, (85, 145, 225), (70, 125, 210))
    return _rotate_img(base, 30)


# (name, image factory, Options) — covers every source path x both modes + rectified.
CASES = [
    ("daikonic", _daikonic_icon, Options()),
    ("daikonic_flatten", _daikonic_icon, Options(flatten=True)),
    ("mastercard", lambda: _synthetic_mastercard()[0], Options()),
    ("mastercard_flatten", lambda: _synthetic_mastercard()[0], Options(flatten=True)),
    ("ring_disk", _ring_disk, Options()),
    ("ring_disk_flatten", _ring_disk, Options(flatten=True)),
    ("linear_grad", _linear_grad, Options()),
    ("linear_grad_flatten", _linear_grad, Options(flatten=True)),
    ("radial_grad", _radial_grad, Options()),
    ("rectified_grad", _rectified_grad, Options()),
    ("rectified_grad_flatten", _rectified_grad, Options(flatten=True)),
]


def _render(name):
    factory, opts = next((f, o) for n, f, o in CASES if n == name)
    return idealize(factory(), options=opts)


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_byte_identical(name):
    golden = (GOLDEN / f"{name}.svg").read_text()
    assert _render(name) == golden, f"{name}: output diverged from golden"
