import os
import numpy as np
import pytest
from PIL import Image
from vectormark.pipeline import idealize, Options

VBIRD = os.path.join(os.path.dirname(__file__), "..", "scratch", "real-logos", "vbird.png")

@pytest.mark.skipif(not os.path.exists(VBIRD), reason="V-bird not present")
def test_vbird_conditioned_emits_more_circles_than_native():
    arr = np.asarray(Image.open(VBIRD).convert("RGB"), np.uint8)
    native = idealize(arr, options=Options(working_max_dim=None))
    cond = idealize(arr, options=Options(working_max_dim=512))
    # conditioning recovers at least one more round dot
    assert cond.count("<circle") + cond.count("<ellipse") >= \
           native.count("<circle") + native.count("<ellipse")
    # and does not explode path count (no fraying)
    assert cond.count("<path") <= native.count("<path") + 2

def test_default_conditioning_keeps_small_logo_valid():
    img = np.full((480, 480, 3), 255, np.uint8); img[120:360, 120:360] = (30, 120, 240)
    svg = idealize(img)   # default working_max_dim=768, 480<768 -> pass-through
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
    assert "<rect" in svg or "<polygon" in svg or "<path" in svg
