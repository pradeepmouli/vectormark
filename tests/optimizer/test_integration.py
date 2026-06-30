from pathlib import Path

import numpy as np
from PIL import Image

from vectormark.optimizer.passes.symmetry import symmetry_pass
from vectormark.pipeline import Options, _optimizer_passes, _prefer_optimizer_svg, idealize


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
    faithful = '<svg><path d="M0 0 Z"/></svg>'
    larger = '<svg><path d="M0 0 Z"/><path d="M1 1 Z"/></svg>'
    longer = '<svg><path d="M0 0 L1 1 L2 2 L3 3 Z"/></svg>'

    assert not _prefer_optimizer_svg(faithful, larger)
    assert not _prefer_optimizer_svg(faithful, longer)
    assert _prefer_optimizer_svg(faithful, faithful)


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

    assert any(token in svg for token in ("<path", "<polygon", "<circle", "<ellipse", "<rect"))
    assert "<use" not in svg


def test_optimizer_pipeline_keeps_daikonic_fixture_objects_present():
    src = Path(__file__).parents[1] / "fixtures" / "daikonic" / "source.png"
    arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)

    svg = idealize(arr, options=Options(optimizer=True, epsilon=1.8, max_error=1.2))

    element_count = sum(svg.count(token) for token in ("<path", "<circle", "<ellipse", "<rect", "<use"))
    assert element_count >= 8
    assert len(svg) > 1000
