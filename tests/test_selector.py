import numpy as np

from vectormark.contour import region_contours
from vectormark.fit import fit_path, recognize_polygon
from vectormark.pipeline import Options
from vectormark.types import Axis, Region
from vectormark.selector import generate_geometry_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_candidates_for_disk_include_circle_and_path_first_is_circle():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    kinds = [c.kind for c in cands]
    assert "circle" in kinds and "path" in kinds
    assert kinds[0] == "circle"          # cascade-priority order: cands[0] == old pick


def test_candidates_nonempty_for_organic_blob():
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 20:60] = True
    mask[20:35, 20:35] = False           # a bite -> not a clean primitive
    region = Region(1, mask, "#222222")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    assert cands and cands[-1].kind == "path"   # fit_path is always the final fallback


def test_straddler_with_symmetric_candidate_excludes_nonsymmetric_fallback():
    # a centered vertical bar (axis x=40): symmetric fits exist, so the non-symmetric
    # recognize_polygon/fit_path fallbacks must NOT be appended (symmetry preserved —
    # the branch's key invariant: no non-symmetric candidate can win over a valid
    # symmetric one for a straddler).
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 30:50] = True
    region = Region(1, mask, "#333333")
    cands = generate_geometry_candidates(region, Options(), Axis(40.0), 2.0)
    assert cands                                  # at least one symmetric candidate exists

    # The bare non-symmetric fit_path fallback (a path with no fill_rule) must be
    # ABSENT from the candidate set: compute what it would be and assert exclusion.
    contour = [c for c in region_contours(mask) if len(c) >= 3][0]
    fp_d = fit_path(contour, epsilon=Options().epsilon, max_error=Options().max_error).params["d"]
    assert not any(c.params.get("d") == fp_d and "fill_rule" not in c.params for c in cands)
    # and no non-symmetric recognize_polygon fallback slipped in either
    poly = recognize_polygon(contour, epsilon=Options().epsilon)
    if poly is not None:
        assert not any(c.kind == "polygon" and c.params.get("points") == poly.params["points"]
                       for c in cands)


from vectormark.selector import select_geometry


def test_select_picks_circle_for_clean_disk():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(50, 50, 32, h, w)
    src[mask] = (30, 100, 235)
    region = Region(1, mask, "#1e64eb")
    shape = select_geometry(region, Options(), None, 0.0, src)
    assert shape is not None and shape.kind == "circle"


def test_select_falls_back_to_first_candidate_without_source():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    shape = select_geometry(region, Options(), None, 0.0, None)
    assert shape is not None and shape.kind == "circle"   # candidates[0]


def test_select_returns_none_when_no_contour():
    region = Region(1, np.zeros((20, 20), bool), "#000000")
    assert select_geometry(region, Options(), None, 0.0, None) is None
