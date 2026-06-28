import numpy as np
from vectormark.pipeline import _segment_image, Options


def _two_blob_img(H=80, W=120):
    img = np.full((H, W, 3), 255, np.uint8)              # white background
    img[20:60, 15:55] = (20, 40, 200)                    # blue block (AA-free synthetic)
    img[20:60, 65:105] = (220, 60, 20)                   # orange block
    return img


def test_segment_attaches_coverage():
    w, h, regions = _segment_image(_two_blob_img(), Options(max_colors=16))
    assert regions, "expected regions"
    for r in regions:
        assert r.coverage is not None
        assert r.coverage.shape == r.mask.shape
        # coverage is ~1 on the region interior, present where the mask is
        assert r.coverage[r.mask].mean() > 0.8


def test_idealize_still_runs_with_coverage():
    from vectormark.pipeline import idealize
    svg = idealize(_two_blob_img(), options=Options(max_colors=16))
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
