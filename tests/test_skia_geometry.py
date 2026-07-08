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
