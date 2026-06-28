"""Bounded-shape-grammar acceptance tests.

V-bird (real logo, ~75 s): verifies the path-count budget holds, no even-odd
speckle holes, and that robust primitive recognition recovers at least one
circle/ellipse for the dot glyphs.

Flat-square (synthetic, <1 s): always-on sanity that the simplest possible
shape stays within budget.

Residual angularity on some paths (staircase AA) is EXPECTED and will be
addressed by the planned feat/aa-contours follow-up; do not tighten bounds
here to fight it.
"""

import os
import re

import numpy as np
import pytest

from vectormark import Options, idealize
from vectormark.fit import MAX_PATH_SEGMENTS

VBIRD = "/Users/pmouli/GitHub.nosync/active/py/vectormark/scratch/real-logos/vbird.png"


def _max_path_segments(svg: str) -> int:
    """Worst-case drawing-command count in any single closed SUBPATH (a path's d may hold
    several M...Z loops for a holed shape; the grammar budget is per simple loop)."""
    worst = 0
    for d in re.findall(r'd="([^"]*)"', svg):
        for sub in re.split(r"(?=M)", d):                 # split into M...  loops
            worst = max(worst, sub.count("L") + sub.count("Q") + sub.count("C"))
    return worst


@pytest.mark.skipif(
    not os.path.exists(VBIRD),
    reason="V-bird image not present in scratch/real-logos/ — skipping real-logo smoke test",
)
def test_vbird_paths_are_bounded_and_unspeckled():
    """Real-logo smoke test: segment budget holds, no even-odd speckle, and at
    least one dot glyph comes through as a circle (robust recognition active)."""
    svg = idealize(VBIRD, options=Options(max_colors=16))

    # Every emitted path fits within the grammar budget (no fraying/noise paths)
    maxseg = _max_path_segments(svg)
    assert maxseg <= MAX_PATH_SEGMENTS, (
        f"V-bird maxseg={maxseg} exceeds grammar budget {MAX_PATH_SEGMENTS}"
    )

    # No even-odd speckle holes: more than one evenodd fill-rule signals a
    # noise-hole leak (one is fine for a legitimate annulus-like shape).
    evenodd_count = svg.count("evenodd")
    assert evenodd_count <= 1, (
        f"V-bird has {evenodd_count} evenodd regions (speckle noise-hole leak)"
    )

    # Robust primitive recognition turned at least one dot into a circle/ellipse.
    assert svg.count("<circle") >= 1 or svg.count("<ellipse") >= 1, (
        "V-bird: no <circle> or <ellipse> emitted — robust recognition did not fire for the dot glyphs"
    )


def test_clean_flat_square_unchanged():
    """Synthetic sanity: a solid red square on white fits the grammar budget
    and emits at least one shape element."""
    img = np.full((80, 80, 3), 255, np.uint8)
    img[20:60, 20:60] = (200, 40, 40)
    svg = idealize(img, options=Options(max_colors=16))
    assert _max_path_segments(svg) <= MAX_PATH_SEGMENTS
    assert "<rect" in svg or "<polygon" in svg or "<path" in svg
