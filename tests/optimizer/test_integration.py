from pathlib import Path

import numpy as np
from PIL import Image

from vectormark.candidate import FlatFill, RadialGradientFill
from vectormark.emit import shape_to_path_d
from vectormark.fit import Shape
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.passes.compound import split_compound_pass
from vectormark.optimizer.passes.primitives import primitives_pass
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.optimizer.passes.symmetry import symmetry_pass
from vectormark.pipeline import Options, _optimizer_passes, _prefer_optimizer_svg, _render_optimizer_body, idealize


def _disk(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _disk_image() -> np.ndarray:
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)
    return img


def _asymmetric_cloud_image() -> np.ndarray:
    img = np.full((96, 128, 3), 255, np.uint8)
    yy, xx = np.ogrid[:96, :128]
    mask = (
        (((xx - 42) ** 2) / 22**2 + ((yy - 52) ** 2) / 18**2 <= 1)
        | (((xx - 66) ** 2) / 30**2 + ((yy - 42) ** 2) / 24**2 <= 1)
        | (((xx - 88) ** 2) / 20**2 + ((yy - 56) ** 2) / 16**2 <= 1)
    )
    mask[58:72, 32:102] = True
    mask[38:48, 88:110] = False
    img[mask] = (20, 70, 150)
    return img


def _two_colored_square_image() -> np.ndarray:
    img = np.full((80, 120, 3), 255, np.uint8)
    img[20:40, 16:36] = (17, 34, 51)
    img[20:40, 70:90] = (171, 205, 239)
    return img


def _synthetic_mastercard_image() -> np.ndarray:
    h, w, gap, r = 240, 360, 58, 104
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((h, w, 3), 255, np.uint8)
    img[left] = (235, 0, 27)
    img[right] = (247, 158, 27)
    img[left & right] = (255, 95, 0)
    return img


def test_optimizer_option_defaults_off_for_existing_pipeline():
    img = _disk_image()

    assert idealize(img, options=Options()) == idealize(img, options=Options(optimizer=False))


def test_optimizer_pipeline_turns_disk_into_circle_and_is_deterministic():
    img = _disk_image()
    opt = Options(optimizer=True)

    first = idealize(img, options=opt)
    second = idealize(img, options=opt)

    assert first == second
    assert "<circle" in first


def test_optimizer_report_includes_object_diagnostics():
    _svg, report = idealize(_disk_image(), options=Options(optimizer=True), report=True)

    data = report.diagnostics.to_dict()
    assert report.elements == 1
    assert report.strategies["optimizer_circle"] == 1
    assert data["optimizer_objects"][0]["shape"] == "circle"


def test_optimizer_structural_fallback_rejects_larger_svg():
    trace = '<svg><path d="M0 0 Z"/></svg>'
    larger = '<svg><path d="M0 0 Z"/><path d="M1 1 Z"/></svg>'
    longer = '<svg><path d="M0 0 L1 1 L2 2 L3 3 Z"/></svg>'

    assert not _prefer_optimizer_svg(trace, larger)
    assert not _prefer_optimizer_svg(trace, longer)
    assert _prefer_optimizer_svg(trace, trace)


def test_optimizer_prefer_allows_small_element_increase_when_bytes_shrink():
    trace = '<svg><path d="M0 0 L10 0 L10 10 L0 10 Z"/></svg>'
    optimized = '<svg><circle r="1"/><use href="#s0"/></svg>'

    assert _prefer_optimizer_svg(trace, optimized)


def test_optimizer_prefer_ignores_non_path_element_count_but_not_path_segments():
    trace = '<svg><path d="M0 0 L10 0 L10 10 L0 10 L0 20 L10 20 L10 30 L0 30 Z"/></svg>'
    many_primitives = '<svg><circle r="1"/><circle r="2"/><use href="#s0"/></svg>'
    simple_long_path = '<svg><path d="M1000.123 1000.456 L2000.789 2000.123 Z"/></svg>'
    more_path_segments = '<svg><path d="M0 0 L1 0 L2 0 L3 0 L4 0 Z"/></svg>'

    assert _prefer_optimizer_svg(trace, many_primitives)
    assert not _prefer_optimizer_svg(simple_long_path, more_path_segments)


def test_optimizer_flatten_emits_paths_without_primitives_or_transforms():
    svg = idealize(_disk_image(), options=Options(optimizer=True, flatten=True))

    assert "<path" in svg
    assert "<circle" not in svg
    assert "<use" not in svg
    assert "transform=" not in svg


def test_optimizer_recolored_clone_keeps_target_fill_without_use_override():
    svg = idealize(_two_colored_square_image(), options=Options(optimizer=True))

    assert "<use" not in svg
    assert 'fill="#112233"' in svg
    assert 'fill="#ABCDEF"' in svg


def test_optimizer_no_symmetry_skips_symmetry_pass():
    assert symmetry_pass in _optimizer_passes(Options(optimizer=True))
    assert symmetry_pass not in _optimizer_passes(Options(optimizer=True, no_symmetry=True))


def test_optimizer_pipeline_does_not_mirror_asymmetric_single_object():
    svg = idealize(_asymmetric_cloud_image(), options=Options(optimizer=True))

    assert "<path" in svg
    assert "<use" not in svg


def test_optimizer_pipeline_keeps_daikonic_fixture_objects_present():
    src = Path(__file__).parents[1] / "fixtures" / "daikonic" / "source.png"
    arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)

    svg = idealize(arr, options=Options(optimizer=True, epsilon=1.8, max_error=1.2))

    element_count = sum(svg.count(token) for token in ("<path", "<circle", "<ellipse", "<rect", "<use"))
    assert element_count >= 8
    assert len(svg) > 1000


def test_optimizer_pipeline_reconstructs_mastercard_as_scene_branch():
    svg, report = idealize(_synthetic_mastercard_image(), options=Options(optimizer=True), report=True)

    assert svg.count("<circle") == 2
    assert svg.count("<path") == 1
    diagnostics = report.diagnostics.to_dict()
    assert diagnostics["optimizer_fallback"] is None
    assert diagnostics["optimizer_regions"][0]["kind"] == "branch"
    assert diagnostics["optimizer_regions"][0]["children"] == 3
    assert [obj["shape"] for obj in diagnostics["optimizer_objects"]] == ["circle", "circle", "path"]


def test_optimizer_inlines_same_gradient_fill_mirror_to_preserve_paint_space():
    fill = RadialGradientFill({"cx": 20.0, "cy": 20.0, "r": 30.0}, [(0.0, "#000000"), (1.0, "#ffffff")])
    source = VectorRegion(
        1,
        Shape("path", {"d": "M10 10 L20 10 L20 30 L10 30 Z"}),
        fill,
        0,
    )
    mirror = VectorRegion(
        2,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 60.0, 0.0)}),
        fill,
        0,
        footprint=source.footprint,
    )

    body, defs = _render_optimizer_body([source, mirror])
    svg_body = "".join(body)

    assert len(defs) == 1
    assert "<use" not in svg_body
    assert svg_body.count("<path") == 2
    assert svg_body.count('fill="url(#g0)"') == 2


def test_optimizer_emits_tree_ids_for_branch_children():
    fill = FlatFill("#111111")
    source = VectorRegion(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        fill,
        0,
    )
    child = VectorRegion(
        3,
        Shape("circle", {"cx": 5.0, "cy": 5.0, "r": 3.0}),
        FlatFill("#FFFFFF"),
        0,
    )
    mirror = VectorRegion(
        4,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0,
        footprint=source.footprint,
    )
    branch = VectorRegion.branch(id=1, children=[source, child, mirror], z=0)

    body, _defs = _render_optimizer_body([branch])
    svg_body = "".join(body)

    assert 'id="s1-1"' in svg_body
    assert 'id="s1-3"' in svg_body
    assert 'href="#s1-1"' not in svg_body
    assert svg_body.count("<path") == 2


def test_optimizer_orders_branch_mirror_before_sibling_covers():
    fill = FlatFill("#111111")
    source = VectorRegion(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        fill,
        0,
    )
    cover = VectorRegion(
        3,
        Shape("circle", {"cx": 5.0, "cy": 5.0, "r": 3.0}),
        FlatFill("#FFFFFF"),
        0.4,
    )
    mirror = VectorRegion(
        5,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0.1,
        footprint=source.footprint,
    )
    branch = VectorRegion.branch(id=1, children=[source, cover, mirror], z=0)

    body, _defs = _render_optimizer_body([branch])

    assert body[0].startswith('<path id="s1-1"')
    assert body[1].startswith('<path')
    assert body[2].startswith('<circle id="s1-3"')


def test_compound_split_turns_cutout_subpaths_into_child_regions():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = shape_to_path_d(Shape("circle", {"cx": 70.0, "cy": 50.0, "r": 12.0}))
    speck = "M10 10 L10.5 10 L10 10.5 Z"
    shape = Shape("path", {"d": f"{outer} {hole} {speck}", "fill_rule": "evenodd"})
    region = VectorRegion(1, shape, FlatFill("#111111"), 0)

    out = optimize(
        [region],
        {region.id: rasterize(region.footprint, (120, 120))},
        [split_compound_pass, primitives_pass],
    )

    assert len(out) == 1
    branch = out[0]
    assert branch.is_branch
    assert len(branch.children) == 2
    assert branch.children[0].fill == region.fill
    assert branch.children[1].fill == FlatFill("#FFFFFF")
    assert branch.children[0].z < branch.children[1].z
    assert branch.children[0].current.kind == "path"
    assert branch.children[1].current.kind == "circle"
