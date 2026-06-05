import numpy as np
from vectormark.contour import outer_contour, rdp, corner_indices, region_contours, _polygon_area


def test_region_contours_finds_hole_outer_first():
    mask = np.zeros((40, 40), bool)
    mask[8:32, 8:32] = True       # square
    mask[16:24, 16:24] = False    # hole / counter
    cs = region_contours(mask)
    assert len(cs) == 2                                  # outer + hole
    assert _polygon_area(cs[0]) > _polygon_area(cs[1])   # outer first


def test_outer_contour_of_square_is_closed_xy():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = outer_contour(mask)
    assert c.ndim == 2 and c.shape[1] == 2          # (N, 2) as (x, y)
    assert np.allclose(c[0], c[-1])                  # closed loop
    assert c[:, 0].min() < 6 and c[:, 0].max() > 13  # spans the square in x


def test_rdp_reduces_straight_run_to_endpoints():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 3]], float)
    simp = rdp(pts, epsilon=0.5)
    assert len(simp) == 3                            # (0,0),(3,0),(3,3)


def test_corner_indices_finds_square_corners():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = rdp(outer_contour(mask), epsilon=1.0)
    corners = corner_indices(c, angle_threshold_deg=45)
    assert len(corners) >= 4


def test_edge_touching_rect_contour_is_closed_and_full():
    mask = np.zeros((20, 20), bool); mask[0:10, 0:8] = True   # touches top + left edges
    c = outer_contour(mask)
    assert np.allclose(c[0], c[-1])                 # closed
    assert c[:, 0].min() <= 0.5 and c[:, 1].min() <= 0.5   # reaches the touched edges
