import skia

from vectormark.skia_geometry import SkPath, path_to_svg_string_faithful


def test_path_to_svg_string_faithful_decomposes_conic_with_requested_pow2():
    path = skia.Path()
    path.moveTo(0, 100)
    path.conicTo(0, 0, 100, 0, 0.70710678)
    path.close()

    d = path_to_svg_string_faithful(path, pow2=3)

    assert d.startswith("M0 100")
    assert d.count("Q") == 8
    assert "C" not in d
    assert d.endswith("Z")
    assert not SkPath.from_svg_d(d).is_empty


def test_distance_measures_between_segments_not_only_vertices():
    left = SkPath(shell=[(0, 0), (2, 0), (2, 10), (0, 10)])
    right = SkPath(shell=[(3, 2), (5, 2), (5, 8), (3, 8)])

    assert left.distance(right) == 1.0


def test_area_subtracts_holes_in_each_disjoint_component():
    first = [(0, 0), (10, 0), (10, 10), (0, 10)]
    first_hole = [(2, 2), (4, 2), (4, 4), (2, 4)]
    second = [(20, 0), (30, 0), (30, 10), (20, 10)]
    second_hole = [(22, 2), (28, 2), (28, 8), (22, 8)]
    path = SkPath(shell=first, holes=[first_hole]).union(SkPath(shell=second, holes=[second_hole]))

    assert path.area == 160.0


def test_symmetric_difference_area_normalizes_overlapping_evenodd_contours():
    left = SkPath(shell=[(0, 0), (10, 0), (10, 10), (0, 10)])
    right = SkPath(shell=[(5, 0), (15, 0), (15, 10), (5, 10)])

    assert left.symmetric_difference(right).area == 100.0
