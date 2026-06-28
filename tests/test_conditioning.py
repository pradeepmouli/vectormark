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
