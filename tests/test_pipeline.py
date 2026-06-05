import numpy as np
from PIL import Image
from vectormark.pipeline import Options, _segment_image, idealize
from tests._render import render_svg, ssim


def _path_points(d):
    import re

    toks = re.findall(r"[MLCQZ]|-?\d*\.?\d+", d)
    counts = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}
    pts, i = [], 0
    while i < len(toks):
        cmd = toks[i]
        i += 1
        n = counts[cmd]
        nums = [float(t) for t in toks[i:i + n]]
        i += n
        pts += [(nums[k], nums[k + 1]) for k in range(0, n, 2)]
    return pts


def test_symmetric_holed_region_emits_exactly_symmetric():
    # a centered annulus (filled disk with a centered hole) is a self-symmetric
    # holed straddler; its idealized geometry should be EXACTLY bilaterally
    # symmetric (built from mirrored halves), not independently-fit contours.
    import re

    h = w = 120
    yy, xx = np.ogrid[:h, :w]
    r2 = (xx - 59.5) ** 2 + (yy - 59.5) ** 2
    mask = (r2 <= 46 ** 2) & (r2 >= 22 ** 2)
    img = np.full((h, w, 3), 255, np.uint8)
    img[mask] = (20, 40, 80)
    svg = idealize(img, options=Options())
    assert "evenodd" in svg                                   # the hole survives

    # every control point's mirror about the axis is also a control point —
    # render-independent proof the geometry is exactly symmetric
    d = re.search(r'\sd="([^"]*)"', svg).group(1)   # \s avoids matching id="s0"
    pts = _path_points(d)
    ax = sum(x for x, _ in pts) / len(pts)
    for x, y in pts:
        assert any(abs(mx - (2 * ax - x)) < 0.6 and abs(my - y) < 0.6 for mx, my in pts)

    # and it still renders recognizably to the source (a thin ring is harsh on
    # SSIM, so this is a sanity floor, not a fidelity assertion)
    assert ssim(render_svg(svg, w, h), img) >= 0.90


def test_area_filter_is_scale_independent():
    # a big block + a small-but-intentional block; padding the canvas with more
    # background must NOT drop the small block (the old canvas-fraction filter did,
    # because its threshold grew with the canvas — it should track the mark).
    # same colour for both (so palette quantization keeps it regardless of how
    # rare the small block becomes); disconnected, so they are two components.
    def scene(pad):
        a = np.full((90 + pad, 90 + pad, 3), 255, np.uint8)
        a[10:60, 10:60] = (0, 0, 0)       # 2500-px block
        a[10:20, 70:80] = (0, 0, 0)       # 100-px block (4% of the largest)
        return a

    tight = _segment_image(scene(0), Options())[2]
    padded = _segment_image(scene(400), Options())[2]   # 4× more background
    assert len(tight) == 2
    assert len(padded) == 2, "small region dropped only because the canvas grew"


def _two_band_logo(path):
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)      # navy rect
    img[34:52, 20:60] = (61, 168, 157)  # teal rect
    Image.fromarray(img).save(path)


def test_idealize_emits_two_rects(tmp_path):
    p = tmp_path / "logo.png"
    _two_band_logo(p)
    svg = idealize(str(p))
    assert svg.count("<rect") == 2
    assert "#062336" in svg and "#3DA89D" in svg
    assert 'viewBox="0 0 80 60"' in svg


def test_solid_color_image_does_not_crash(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((24, 24, 3), (40, 40, 40), np.uint8)).save(tmp_path / "solid.png")
    svg = idealize(str(tmp_path / "solid.png"))
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")

def test_gradient_image_does_not_crash():
    grad = np.zeros((40, 40, 3), np.uint8)
    for x in range(40):
        grad[:, x] = (x * 6, 100, 255 - x * 6)
    svg = idealize(grad)
    assert svg.startswith("<svg")


def test_region_with_hole_uses_evenodd_and_renders_hole():
    from tests._render import render_svg
    img = np.full((60, 60, 3), 255, np.uint8)
    img[10:50, 10:50] = (6, 35, 54)      # navy square
    img[24:36, 24:36] = (255, 255, 255)  # white counter (hole)
    svg = idealize(img)
    assert 'fill-rule="evenodd"' in svg
    out = render_svg(svg, 60, 60)
    assert out[30, 30].min() > 200       # hole renders as background (light)
    assert int(out[15, 30].sum()) < 200  # frame renders navy (dark)


def test_flatten_emits_only_paths():
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)
    img[34:52, 20:60] = (61, 168, 157)
    flat = idealize(img, options=Options(flatten=True))
    assert "<path" in flat
    for elem in ("<rect", "<ellipse", "<circle", "<polygon", "<use"):
        assert elem not in flat


def _tapered_band_img(H=90, W=120, axis=60):
    img = np.full((H, W, 3), 255, np.uint8)
    m = np.zeros((H, W), bool)
    for y in range(20, 70):
        hw = int(40 - (y - 20) * 0.2)        # straight taper -> rounded_trapezoid path, not <rect>
        m[y, axis - hw:axis + hw] = True
    img[m] = (6, 35, 54)
    return img


def test_corner_radius_override_changes_output():
    img = _tapered_band_img()
    sharp = idealize(img, options=Options(corner_radius=0.0))
    rounded = idealize(img, options=Options(corner_radius=10.0))
    assert "<rect" not in rounded                  # tapered band -> rounded-trapezoid path
    assert sharp != rounded                        # the shared radius actually drives geometry
