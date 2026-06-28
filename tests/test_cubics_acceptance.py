import re, os
import numpy as np
import pytest
from PIL import Image
from vectormark._fitcurve import fit_cubic_beziers, cubic_inflects
from vectormark.pipeline import idealize, Options

def test_random_convex_and_concave_runs_never_inflect_after_fit():
    rng_pts = [
        np.c_[np.linspace(0, 100, 30), 40*np.sin(np.linspace(0, np.pi, 30))],      # convex bump
        np.c_[np.linspace(0, 100, 30), -30*np.sin(np.linspace(0, np.pi, 30))],     # concave
        np.c_[np.linspace(0, 120, 40), 50*np.sin(np.linspace(0, 2*np.pi, 40))],    # S (must split)
    ]
    for pts in rng_pts:
        for c in fit_cubic_beziers(pts, max_error=1.5):
            assert not cubic_inflects(c)

VBIRD = os.path.join(os.path.dirname(__file__), "..", "scratch", "real-logos", "vbird.png")

@pytest.mark.skipif(not os.path.exists(VBIRD), reason="V-bird not present")
def test_vbird_conditioned_uses_cubics_and_stays_bounded():
    arr = np.asarray(Image.open(VBIRD).convert("RGB"), np.uint8)
    svg = idealize(arr, options=Options(working_max_dim=512))
    assert "C" in svg                                  # cubics in real output
    # no single subpath is a frayed mega-trace
    worst = max((sum(sub.count(ch) for ch in "LCQ") for d in re.findall(r'd="([^"]*)"', svg)
                 for sub in re.split(r"(?=M)", d)), default=0)
    assert worst <= 12
