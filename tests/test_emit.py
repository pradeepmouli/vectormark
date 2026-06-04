from vectormark.fit import Shape
from vectormark.types import Axis
from vectormark.emit import shape_to_svg, mirror_use, render_svg_doc


def test_rect_shape_emits_native_rect():
    s = Shape("rect", {"x": 5, "y": 6, "w": 20, "h": 10})
    out = shape_to_svg(s, fill="#3DA89D", elem_id="r1")
    assert out.startswith("<rect") and 'width="20"' in out and 'fill="#3DA89D"' in out


def test_circle_and_ellipse_and_polygon():
    assert "<circle" in shape_to_svg(Shape("circle", {"cx": 5, "cy": 5, "r": 4}), "#000", "c")
    assert "<ellipse" in shape_to_svg(Shape("ellipse", {"cx": 5, "cy": 5, "rx": 4, "ry": 3}), "#000", "e")
    poly = shape_to_svg(Shape("polygon", {"points": [(0, 0), (4, 0), (2, 4)]}), "#000", "p")
    assert "<polygon" in poly and "points=" in poly


def test_mirror_use_reflects_about_axis():
    use = mirror_use("leaf1", Axis(x=60.0))
    # reflection about x=a is matrix(-1 0 0 1 2a 0)
    assert 'href="#leaf1"' in use and "matrix(-1 0 0 1 120 0)" in use


def test_render_svg_doc_wraps_with_viewbox():
    doc = render_svg_doc(120, 100, ['<rect x="0" y="0" width="1" height="1" fill="#000"/>'])
    assert 'viewBox="0 0 120 100"' in doc and doc.strip().endswith("</svg>")
