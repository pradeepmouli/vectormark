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


def test_idealize_is_deterministic_bit_identical():
    # The AA-coverage path must be deterministic — same bytes in, byte-identical SVG out.
    from vectormark.pipeline import idealize, Options
    img = _two_blob_img()
    a = idealize(img, options=Options(max_colors=16))
    b = idealize(img, options=Options(max_colors=16))
    assert a == b


def test_thin_feature_survives_coverage_path():
    # A thin bar (a few px wide) must still produce a region/candidate, not vanish, when the
    # soft-coverage contour path is active.
    import numpy as np
    from vectormark.pipeline import _segment_image, Options
    img = np.full((80, 120, 3), 255, np.uint8)
    img[20:60, 58:62] = (20, 40, 200)          # 4px-wide vertical bar
    _, _, regions = _segment_image(img, Options(max_colors=16))
    assert any(r.mask.sum() > 50 for r in regions), "thin bar must survive segmentation"


def test_three_region_junction_produces_valid_svg():
    # Triple-point junction (the documented Strategy-2 risk is DEFERRED, but the pipeline must
    # not crash or emit malformed output there). Three colors meeting at a point.
    import numpy as np
    from vectormark.pipeline import idealize, Options
    img = np.full((90, 90, 3), 255, np.uint8)
    img[10:80, 10:45] = (200, 40, 40)          # left
    img[10:45, 45:80] = (40, 160, 40)          # top-right
    img[45:80, 45:80] = (40, 40, 200)          # bottom-right  (all meet near (45,45))
    svg = idealize(img, options=Options(max_colors=16))
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
    assert svg.count("<path") + svg.count("<polygon") + svg.count("<rect") >= 2
