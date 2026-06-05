import numpy as np

from vectormark.contour import region_contours
from vectormark.emit import path_svg, render_svg_doc
from vectormark.refine import symmetric_fit
from tests._render import render_svg, ssim


def _fold_ssim(img, axis_x):
    w = img.shape[1]
    k = min(axis_x, w - axis_x)
    return float(ssim(img[:, axis_x - k:axis_x][:, ::-1], img[:, axis_x:axis_x + k]))


def test_symmetric_fit_dome_is_exactly_symmetric():
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)  # half-ellipse
    cs = region_contours(dome)
    sh = symmetric_fit(cs[0], 40.0, epsilon=1.5, max_error=1.0)
    assert sh is not None and sh.kind == "path"
    img = render_svg(render_svg_doc(W, H, [path_svg(sh.params["d"], "#062336")]), W, H)
    assert _fold_ssim(img, 40) >= 0.999          # exactly symmetric about the axis


def _tapered_band(H=60, W=120):
    m = np.zeros((H, W), bool)
    for y in range(12, 48):
        hw = int(46 - (y - 12) * 0.2)     # straight taper
        m[y, 60 - hw:60 + hw] = True
    return m


def test_rounded_trapezoid_fit_symmetric_and_rejects_curves():
    from vectormark.refine import rounded_trapezoid_fit
    c = region_contours(_tapered_band())[0]
    sh = rounded_trapezoid_fit(c, 60.0, max_error=1.5)
    assert sh is not None and sh.kind == "path"
    img = render_svg(render_svg_doc(120, 60, [path_svg(sh.params["d"], "#000")]), 120, 60)
    assert _fold_ssim(img, 60) >= 0.999            # exactly symmetric

    # a curved half-ellipse is NOT a trapezoid -> None (falls through to symmetric_fit)
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    assert rounded_trapezoid_fit(region_contours(dome)[0], 40.0, max_error=1.0) is None


def test_symmetric_fit_beats_raw_path_symmetry():
    from vectormark.fit import fit_path
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    c = region_contours(dome)[0]
    sym = render_svg(render_svg_doc(W, H, [path_svg(symmetric_fit(c, 40.0, epsilon=1.5, max_error=1.0).params["d"], "#000")]), W, H)
    raw = render_svg(render_svg_doc(W, H, [path_svg(fit_path(c, epsilon=1.5, max_error=1.0).params["d"], "#000")]), W, H)
    assert _fold_ssim(sym, 40) > _fold_ssim(raw, 40)
