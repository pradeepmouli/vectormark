import numpy as np
from vectormark.pipeline import _condition_input, Options

def test_passthrough_when_within_threshold():
    arr = np.zeros((400, 300, 3), np.uint8)
    out = _condition_input(arr, 768)
    assert out is arr or (out.shape == arr.shape and np.array_equal(out, arr))

def test_passthrough_when_disabled():
    arr = np.zeros((2000, 1000, 3), np.uint8)
    assert np.array_equal(_condition_input(arr, None), arr)

def test_downscales_longest_side_to_target():
    arr = np.zeros((1500, 900, 3), np.uint8)
    out = _condition_input(arr, 768)
    assert max(out.shape[:2]) == 768
    assert out.shape[:2] == (768, round(900 * 768 / 1500))   # aspect preserved
    assert out.dtype == np.uint8

def test_deterministic():
    rng = np.zeros((1200, 1200, 3), np.uint8); rng[300:900, 300:900] = (20, 40, 200)
    assert np.array_equal(_condition_input(rng, 512), _condition_input(rng, 512))

def test_default_option_is_768():
    assert Options().working_max_dim == 768

import re
from vectormark.pipeline import idealize

def _big_logo(H=1500, W=1500):
    img = np.full((H, W, 3), 255, np.uint8)
    img[400:1100, 400:1100] = (200, 40, 40)     # big red square
    return img

def test_output_preserves_original_dimensions():
    svg = idealize(_big_logo(), options=Options(working_max_dim=768))
    m = re.search(r'<svg[^>]*\bwidth="(\d+)"\s+height="(\d+)"', svg)
    assert m and (int(m.group(1)), int(m.group(2))) == (1500, 1500)
    vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert vb and max(int(vb.group(1)), int(vb.group(2))) == 768   # working space

def test_small_input_identical_to_disabled():
    img = np.full((300, 300, 3), 255, np.uint8); img[80:220, 80:220] = (20, 40, 200)
    assert idealize(img, options=Options(working_max_dim=768)) == \
           idealize(img, options=Options(working_max_dim=None))

def test_conditioned_idealize_is_deterministic():
    big = _big_logo()
    assert idealize(big, options=Options(working_max_dim=512)) == \
           idealize(big, options=Options(working_max_dim=512))
