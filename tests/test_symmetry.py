import numpy as np
from vectormark.types import Region
from vectormark.symmetry import detect_axis, classify_regions


def _sym_masks():
    H, W = 40, 40
    dome = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    dome[((xx - 20) ** 2 / 100 + (yy - 20) ** 2 / 64) <= 1] = True  # centered ellipse
    left = np.zeros((H, W), bool); left[5:10, 6:12] = True
    right = np.zeros((H, W), bool); right[5:10, 29:35] = True       # mirror of left[6:12) about x=20
    return [Region(1, dome, "#062336"), Region(2, left, "#062336"), Region(3, right, "#062336")]


def test_detect_axis_finds_center():
    regions = _sym_masks()
    union = np.any([r.mask for r in regions], axis=0)
    axis = detect_axis(union)
    assert axis is not None
    assert abs(axis.x - 19.5) < 1.0  # image center for W=40 is 19.5


def test_classify_straddle_vs_pair():
    regions = _sym_masks()
    axis = detect_axis(np.any([r.mask for r in regions], axis=0))
    straddlers, pairs, loners = classify_regions(regions, axis)
    assert len(straddlers) == 1 and straddlers[0].label == 1       # the dome
    assert len(pairs) == 1                                          # left+right as one pair
    assert loners == []                                            # nothing asymmetric+unpaired
    canon, _mirror = pairs[0]
    assert {canon.label, _mirror.label} == {2, 3}


def test_classify_isolates_lone_asymmetric_region():
    # an asymmetric region with no mirror partner must NOT be treated as a
    # self-symmetric straddler (else _fit_region would force-mirror it).
    regions = _sym_masks()
    asym = np.zeros((40, 40), bool)
    asym[6:10, 6:9] = True
    asym[8:10, 6:16] = True                                        # an L — not self-symmetric
    regions.append(Region(4, asym, "#ff0000"))                    # unique colour ⇒ no partner
    axis = detect_axis(np.any([r.mask for r in regions], axis=0))
    straddlers, pairs, loners = classify_regions(regions, axis)
    assert [r.label for r in loners] == [4]
    assert 4 not in [r.label for r in straddlers]


def test_no_symmetry_returns_none():
    # a centered square IS bilaterally symmetric — knock out one corner so the
    # shape has no vertical axis at all
    asym = np.zeros((30, 30), bool); asym[2:8, 2:8] = True
    asym[2:4, 2:4] = False
    assert detect_axis(asym) is None


def test_axis_reachable_for_fractional_centroid():
    # symmetric about x=19.5 but with an off-grid centroid
    m = np.zeros((30, 40), bool)
    m[5:25, 12:28] = True          # symmetric block about 19.5
    m[5:8, 12:14] = True           # nudge centroid off .5 grid (still ~symmetric enough)
    ax = detect_axis(m)
    assert ax is not None and abs(ax.x - 19.5) <= 0.5
