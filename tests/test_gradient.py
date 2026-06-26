# SPDX-License-Identifier: MIT
import numpy as np


def _OKLAB(img_uint8):
    from vectormark.color import srgb_to_oklab
    return srgb_to_oklab(img_uint8.reshape(-1, 3) / 255.0)


def test_mean_delta_e_zero_for_identical_and_positive_for_different():
    from vectormark.color import mean_delta_e
    a = np.full((4, 4, 3), 120, np.uint8)
    assert mean_delta_e(a, a) == 0.0
    b = a.copy()
    b[:, :, 0] = 200          # shift red channel
    assert mean_delta_e(a, b) > 0.02


def test_linear_gradient_def_emits_stops_and_coords():
    from vectormark.emit import linear_gradient_def
    d = linear_gradient_def("g0", 10, 20, 110, 20, [(0.0, "#ff0000"), (1.0, "#0000ff")])
    assert d.startswith("<linearGradient") and 'id="g0"' in d
    assert 'gradientUnits="userSpaceOnUse"' in d
    assert 'x1="10"' in d and 'x2="110"' in d
    assert d.count("<stop") == 2
    assert 'offset="0"' in d and 'stop-color="#ff0000"' in d
    assert 'offset="1"' in d and 'stop-color="#0000ff"' in d


def test_radial_gradient_def_emits_center_radius():
    from vectormark.emit import radial_gradient_def
    d = radial_gradient_def("g1", 50, 60, 40, [(0.0, "#ffffff"), (1.0, "#000000")])
    assert d.startswith("<radialGradient") and 'id="g1"' in d
    assert 'cx="50"' in d and 'cy="60"' in d and 'r="40"' in d
    assert d.count("<stop") == 2


def test_render_svg_doc_wraps_defs():
    from vectormark.emit import render_svg_doc
    out = render_svg_doc(100, 100, ['<rect/>'], defs=['<linearGradient id="g0"></linearGradient>'])
    assert "<defs>" in out and "</defs>" in out
    assert out.index("<defs>") < out.index("<rect/>")     # defs before body
    out2 = render_svg_doc(100, 100, ['<rect/>'])
    assert "<defs>" not in out2                            # no defs block when none given


def _hstrip_regions(colors_hex, h=40, band_w=12):
    """Adjacent vertical bands left→right, one per color (a quantized ramp)."""
    from vectormark.types import Region
    w = band_w * len(colors_hex)
    regions = []
    for i, c in enumerate(colors_hex):
        m = np.zeros((h, w), bool)
        m[:, i * band_w:(i + 1) * band_w] = True
        regions.append(Region(label=i + 1, mask=m, color_hex=c))
    return regions


def _linear_gradient_image(h, w, p0, p1, stops_rgb):
    """Render a ground-truth linear gradient (for fitting against)."""
    yy, xx = np.mgrid[:h, :w]
    d = np.array(p1, float) - np.array(p0, float)
    L = np.dot(d, d)
    t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb])
    cols = np.array([c for _, c in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_linear_recovers_axis_and_endpoints():
    from vectormark.gradient import _fit_linear
    h, w = 80, 120
    p0, p1 = (10, 40), (110, 40)                       # horizontal axis
    img = _linear_gradient_image(h, w, p0, p1, [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    ys, xs = np.mgrid[:h, :w]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    oklab = _OKLAB(img)
    model = _fit_linear(pts, oklab, img.reshape(-1, 3))
    assert model is not None and model["kind"] == "linear"
    g = model["geometry"]
    # axis is ~horizontal: endpoints span x, ~constant y
    assert abs(g["y1"] - g["y2"]) < 3.0
    # span tolerance is loose: the test gradient fills the whole 120px canvas, so the
    # fitted axis spans all pixels (~119px) not the 100px p0->p1 range; the DIRECTION
    # assertion above (|y1-y2|) is the real check. (In the pipeline the footprint mask
    # bounds the span.)
    assert abs(abs(g["x2"] - g["x1"]) - 100) < 25
    assert len(model["stops"]) >= 2


def test_reduce_stops_drops_redundant_midpoints():
    from vectormark.gradient import _reduce_stops
    # an sRGB gray ramp: some interior stops fall within max_delta_e of the
    # piecewise-linear (OKLab) reconstruction from their neighbors, so they get dropped
    stops = [(0.0, "#000000"), (0.25, "#404040"), (0.5, "#808080"),
             (0.75, "#bfbfbf"), (1.0, "#ffffff")]
    reduced = _reduce_stops(stops, max_delta_e=0.02)
    assert len(reduced) < len(stops) and reduced[0][0] == 0.0 and reduced[-1][0] == 1.0


def _radial_gradient_image(h, w, c, r, stops_rgb):
    yy, xx = np.mgrid[:h, :w]
    t = (np.hypot(xx - c[0], yy - c[1]) / r).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([col for _, col in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_radial_recovers_center():
    from vectormark.gradient import _fit_radial
    h, w = 120, 120
    c, r = (60, 60), 50
    img = _radial_gradient_image(h, w, c, r, [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])
    ys, xs = np.mgrid[:h, :w]
    # restrict to the disc so background doesn't dominate the fit
    inside = np.hypot(xs - c[0], ys - c[1]) <= r
    pts = np.column_stack([xs[inside], ys[inside]]).astype(float)
    oklab = _OKLAB(img[inside])
    model = _fit_radial(pts, oklab, img[inside].reshape(-1, 3))
    assert model is not None and model["kind"] == "radial"
    g = model["geometry"]
    assert abs(g["cx"] - 60) < 6 and abs(g["cy"] - 60) < 6
    assert g["r"] > 30


def test_fit_gradient_accepts_linear_rejects_flat():
    from vectormark.gradient import fit_gradient
    h, w = 80, 120
    img = _linear_gradient_image(h, w, (10, 40), (110, 40),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    mask = np.ones((h, w), bool)
    model = fit_gradient(mask, img)
    assert model is not None and model["kind"] == "linear"
    flat = np.full((h, w, 3), (37, 99, 235), np.uint8)
    assert fit_gradient(mask, flat) is None             # flat -> no gradient


def test_detect_gradients_consumes_ramp_returns_remaining():
    from vectormark.gradient import detect_gradients
    from vectormark.types import Region
    h, w = 60, 160
    # left half: a 10-band blue->magenta linear ramp (8px each); right: one flat green block.
    # 10 thin bands keep each band at ~5% of total_fg, below _THIN_BAND_TOL so the group
    # passes the fillability gate (representative of real gradient quantization).
    img = _linear_gradient_image(h, 80, (0, 30), (79, 30),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    full = np.zeros((h, w, 3), np.uint8)
    full[:, :80] = img
    full[:, 80:] = (20, 160, 60)
    # build the quantized regions the way the pipeline would (10 ramp bands + 1 flat)
    regions = []
    for i in range(10):
        m = np.zeros((h, w), bool); m[:, i * 8:(i + 1) * 8] = True
        regions.append(Region(label=i + 1, mask=m,
                              color_hex="#%02x%02x%02x" % tuple(np.median(full[m], axis=0).astype(int))))
    gm = np.zeros((h, w), bool); gm[:, 80:] = True
    regions.append(Region(label=11, mask=gm, color_hex="#149c3c"))
    fills, remaining = detect_gradients(regions, full)
    assert len(fills) == 1                               # the ramp became one gradient fill
    assert fills[0][1]["kind"] == "linear"
    assert {r.label for r in remaining} == {11}          # the flat green block remains


def test_expand_footprint_bounds_to_contiguous_region():
    from vectormark.gradient import _expand_footprint
    h, w = 40, 120
    # horizontal linear model the image matches everywhere EXCEPT a non-matching black gap
    model = {"kind": "linear", "geometry": {"x1": 0.0, "y1": 20.0, "x2": 119.0, "y2": 20.0},
             "stops": [(0.0, "#2563eb"), (1.0, "#db2777")]}
    ys, xs = np.mgrid[:h, :w]
    from vectormark.gradient import _model_t, _interp_stops_rgb
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    img = _interp_stops_rgb(_model_t(model, pts), model["stops"]).reshape(h, w, 3).round().astype(np.uint8)
    img[:, 60:71] = (0, 0, 0)                       # non-matching black gap breaks contiguity
    mask = np.zeros((h, w), bool); mask[:, 0:50] = True   # surviving bands (left of the gap)
    expanded = _expand_footprint(model, mask, img)
    assert expanded[:, 50:60].all()                # contiguous matching strip recovered
    assert not expanded[:, 71:].any()              # matching pixels PAST the gap stay out (bounded)
    assert not expanded[:, 60:71].any()            # the black gap itself is never absorbed


def test_detect_gradients_zigzag_bands_stay_flat():
    from vectormark.color import oklab_to_srgb, srgb_to_oklab
    from vectormark.gradient import detect_gradients
    from vectormark.types import Region
    # Bands whose colours zig-zag along a line in OKLab so every spatially-adjacent pair is
    # a LARGE colour step -> merge_components never joins them -> all stay flat in `remaining`.
    l0 = srgb_to_oklab(np.array([20, 50, 250])[None] / 255.0)[0]
    l1 = srgb_to_oklab(np.array([250, 20, 90])[None] / 255.0)[0]

    def hex_at(o):
        rgb = (np.clip(oklab_to_srgb((l0 + o * (l1 - l0))[None])[0], 0, 1) * 255).round().astype(int)
        return "#%02x%02x%02x" % tuple(rgb)

    spatial = [0.0, 1.0, 0.2, 0.8, 0.4, 0.6]
    h, band_w = 40, 18
    w = band_w * len(spatial)
    img = np.zeros((h, w, 3), np.uint8)
    regions = []
    for i, o in enumerate(spatial):
        hx = hex_at(o)
        m = np.zeros((h, w), bool)
        m[:, i * band_w:(i + 1) * band_w] = True
        img[m] = (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))
        regions.append(Region(label=i + 1, mask=m, color_hex=hx))
    fills, remaining = detect_gradients(regions, img)
    assert fills == []
    assert {r.label for r in remaining} == {1, 2, 3, 4, 5, 6}


def test_dominant_blob_fraction():
    from vectormark.gradient import _dominant_blob_fraction
    m = np.zeros((20, 40), bool)
    m[2:18, 2:18] = True                       # one 16x16 blob, rest empty
    assert _dominant_blob_fraction(m) == 1.0
    m[2:18, 22:38] = True                      # add a second, equal, disconnected blob
    assert abs(_dominant_blob_fraction(m) - 0.5) < 1e-9
    assert _dominant_blob_fraction(np.zeros((5, 5), bool)) == 0.0   # empty -> 0


def _smooth_linear_region(h, w, c0, c1):
    """One full-canvas Region + a horizontally smooth linear gradient image (raw pixels)."""
    from vectormark.types import Region
    yy, xx = np.mgrid[:h, :w]
    t = xx / (w - 1)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = c0[ch] + t * (c1[ch] - c0[ch])
    img = img.round().astype(np.uint8)
    return [Region(label=1, mask=np.ones((h, w), bool), color_hex="#000000")], img


def test_detect_gradients_smooth_single_blob_fits():
    from vectormark.gradient import detect_gradients
    regions, img = _smooth_linear_region(60, 120, (20, 40, 200), (220, 40, 90))
    fills, remaining = detect_gradients(regions, img)
    assert len(fills) == 1 and fills[0][1]["kind"] == "linear"
    assert remaining == []                                   # the blob was consumed


def test_detect_gradients_smooth_rejects_multiblob():
    from vectormark.types import Region
    from vectormark.gradient import detect_gradients
    h, w = 60, 140
    img = np.full((h, w, 3), 255, np.uint8)
    img[10:50, 10:50] = (220, 30, 30)                        # two disconnected flat blocks
    img[10:50, 90:130] = (30, 30, 220)
    m1 = np.zeros((h, w), bool); m1[10:50, 10:50] = True
    m2 = np.zeros((h, w), bool); m2[10:50, 90:130] = True
    regions = [Region(label=1, mask=m1, color_hex="#dc1e1e"),
               Region(label=2, mask=m2, color_hex="#1e1edc")]
    fills, remaining = detect_gradients(regions, img)
    assert fills == [] and {r.label for r in remaining} == {1, 2}   # dom 0.5 < 0.85


def test_detect_gradients_smooth_rejects_flat_blob():
    from vectormark.types import Region
    from vectormark.gradient import detect_gradients
    h, w = 50, 50
    img = np.full((h, w, 3), (40, 120, 200), np.uint8)       # one flat colour
    regions = [Region(label=1, mask=np.ones((h, w), bool), color_hex="#2878c8")]
    fills, remaining = detect_gradients(regions, img)
    assert fills == [] and {r.label for r in remaining} == {1}   # fit_gradient -> None


def test_fit_gradient_rejects_near_flat_region():
    # a near-flat red region with only a faint ~2-level tint (like a real flat logo's
    # antialiasing/compression noise): its fitted stops barely travel, so it must NOT be
    # emitted as a near-constant gradient (regression for Pinterest/Vimeo over-detection).
    from vectormark.gradient import fit_gradient
    h, w = 60, 60
    img = np.empty((h, w, 3))
    t = np.linspace(0.0, 1.0, w)
    for ch, (a, b) in enumerate(((189, 191), (8, 10), (28, 30))):   # #BD081C +/- ~2 levels
        img[:, :, ch] = a + t * (b - a)
    img = img.round().astype(np.uint8)
    assert fit_gradient(np.ones((h, w), bool), img) is None         # span below _MIN_STOP_SPAN


def test_fit_gradient_accepts_traveling_gradient():
    from vectormark.gradient import fit_gradient
    h, w = 60, 120
    img = np.empty((h, w, 3))
    t = np.linspace(0.0, 1.0, w)
    for ch, (a, b) in enumerate(((37, 219), (99, 39), (235, 119))):  # blue -> magenta
        img[:, :, ch] = a + t * (b - a)
    img = img.round().astype(np.uint8)
    model = fit_gradient(np.ones((h, w), bool), img)
    assert model is not None and model["kind"] == "linear"          # real travel: still fires


def test_best_parametric_searched_beats_heuristic_on_offcenter_radial():
    # a radial gradient whose centre is in a corner (where the principal-axis-extreme
    # heuristic lands poorly); the searched fit must find a low-mean-ΔE radial model.
    from vectormark.gradient import _best_parametric
    h, w = 80, 80
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - 5, yy - 5) / np.hypot(w, h)        # centre near (5,5) corner
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((230, 30), (120, 60), (40, 210))):
        img[:, :, ch] = (a + r * (b - a))
    img = img.round().astype(np.uint8)
    out = _best_parametric(np.ones((h, w), bool), img)
    assert out is not None
    model, mean_de, median_de = out
    assert model["kind"] in ("radial", "linear")
    assert mean_de < 0.05 and median_de < 0.05          # a real gradient fits tightly


def test_best_parametric_returns_none_for_flat():
    from vectormark.gradient import _best_parametric
    img = np.full((40, 40, 3), (50, 100, 150), np.uint8)
    assert _best_parametric(np.ones((40, 40), bool), img) is None   # span below minimum


def _diagonal_2d_field(h, w):
    """A separable 2-D field (horizontal hue x vertical luminance) that NO single
    linear/radial gradient fits: hue runs left->right, brightness runs top->bottom."""
    yy, xx = np.mgrid[:h, :w]
    tx = xx / (w - 1)
    ty = yy / (h - 1)
    img = np.empty((h, w, 3))
    img[:, :, 0] = 30 + tx * 200                       # R climbs with x
    img[:, :, 1] = 20 + ty * 200                       # G climbs with y
    img[:, :, 2] = 200 - tx * 160                      # B falls with x
    return img.round().astype(np.uint8)


def test_fit_stretch_returns_raster_model_under_target():
    from vectormark.gradient import _fit_stretch, _STRETCH_TARGET
    img = _diagonal_2d_field(96, 96)
    model = _fit_stretch(np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"
    g = model["geometry"]
    assert (g["x"], g["y"], g["w"], g["h"]) == (0.0, 0.0, 96.0, 96.0)
    assert isinstance(model["png_b64"], str) and len(model["png_b64"]) > 0


def test_fit_stretch_none_for_degenerate_bbox():
    from vectormark.gradient import _fit_stretch
    m = np.zeros((40, 40), bool)
    m[10, 5:9] = True                                  # 1px tall footprint
    assert _fit_stretch(m, np.zeros((40, 40, 3), np.uint8)) is None


def test_merge_components_merges_small_steps_into_one():
    from vectormark.gradient import merge_components
    # 4 adjacent bands stepping blue->magenta (small OKLab steps between neighbours)
    regions = _hstrip_regions(["#2563eb", "#7b3fc4", "#b13a9e", "#db2777"])
    groups = merge_components(regions, tol=0.15)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_merge_components_splits_at_large_step():
    from vectormark.gradient import merge_components
    # a small-step pair, then a large jump to a distinct hue, then another small-step pair
    regions = _hstrip_regions(["#2563eb", "#3a6ae0", "#11aa33", "#15b53a"])
    groups = merge_components(regions, tol=0.15)
    labels = sorted(sorted(r.label for r in g) for g in groups)
    assert labels == [[1, 2], [3, 4]]                 # split at the blue->green jump


def test_merge_components_singleton_when_isolated_by_large_steps():
    from vectormark.gradient import merge_components
    # zig-zag hues: every adjacency is a large step -> no merges -> all singletons
    regions = _hstrip_regions(["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
    groups = merge_components(regions, tol=0.15)
    assert sorted(len(g) for g in groups) == [1, 1, 1, 1]


def test_merge_components_transitive_chain():
    from vectormark.gradient import merge_components
    # a long chain of small steps merges end-to-end even though the ends are far apart
    regions = _hstrip_regions(["#2563eb", "#5a4fd0", "#8a44b4", "#b13a9e", "#db2777"])
    groups = merge_components(regions, tol=0.15)
    assert len(groups) == 1 and len(groups[0]) == 5


def _2d_field(h, w):
    """A smooth field that no single linear/radial gradient fits under the param bound:
    horizontal hue ramp plus a contrasting corner."""
    yy, xx = np.mgrid[:h, :w]
    t = xx / (w - 1)
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((30, 230), (60, 60), (220, 40))):
        img[:, :, ch] = a + t * (b - a)
    img[(xx >= w * 0.5) & (yy >= h * 0.5)] = (20, 230, 40)
    return img.round().astype(np.uint8)


def test_component_fill_strict_gradient_for_clean_ramp():
    from vectormark.gradient import _component_fill
    h, w = 60, 120
    img = _linear_gradient_image(h, w, (0, 30), (119, 30),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    model = _component_fill(np.ones((h, w), bool), img)
    assert model is not None and model["kind"] in ("linear", "radial")


def test_component_fill_none_for_flat():
    from vectormark.gradient import _component_fill
    img = np.full((40, 40, 3), (50, 100, 150), np.uint8)
    assert _component_fill(np.ones((40, 40), bool), img) is None     # flat -> solid colour


def test_component_fill_raster_for_2d_field():
    from vectormark.gradient import _component_fill
    img = _2d_field(96, 96)
    model = _component_fill(np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"


def _regions_with_areas(areas):
    from vectormark.types import Region
    out = []
    for i, a in enumerate(areas):
        m = np.zeros((1, 1000), bool); m[0, :a] = True       # a True pixels -> Region.area == a
        out.append(Region(label=i + 1, mask=m, color_hex="#000000"))
    return out


def test_group_is_fillable_dominant_thin_chunky():
    from vectormark.gradient import _group_is_fillable
    # dominant single blob (90% of fg): needs within-region variation >= _SMOOTH_VAR_TOL.
    # Build a smoothly-varying ramp over the 900-pixel region so it qualifies as a
    # genuine continuous tone (not a flat dominant blob like a two-tone logo).
    img_shape = (1, 1000, 3)
    img_ramp = np.zeros(img_shape, np.uint8)
    img_ramp[0, :900, :] = np.linspace(0, 200, 900, dtype=np.uint8)[:, None]
    assert _group_is_fillable(_regions_with_areas([900]), 1000.0, img_ramp) is True
    # 10 thin bands, 80% of fg, each 8% -> avg 0.08 < 0.10 -> finely-quantized -> fillable
    # (thin-bands path does not use within-region variation; any image works)
    img_flat = np.zeros(img_shape, np.uint8)
    assert _group_is_fillable(_regions_with_areas([80] * 10), 1000.0, img_flat) is True
    # 4 chunky facets, 80% of fg, each 20% -> avg 0.20 >= 0.10 and not dominant -> NOT fillable
    assert _group_is_fillable(_regions_with_areas([200] * 4), 1000.0, img_flat) is False


def test_group_is_fillable_rejects_dominant_two_tone_flats():
    # two adjacent internally-flat regions that together dominate the foreground and whose
    # colours are within MERGE_TOL must NOT be gradient-fillable (a crisp two-tone logo stays flat).
    from vectormark.gradient import _group_is_fillable
    from vectormark.types import Region
    h, w = 120, 200
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :w // 2] = (60, 90, 200)
    img[:, w // 2:] = (92, 128, 202)                 # OKLab step ~0.11 (< MERGE_TOL, > _MIN_STOP_SPAN)
    m1 = np.zeros((h, w), bool); m1[:, :w // 2] = True
    m2 = np.zeros((h, w), bool); m2[:, w // 2:] = True
    group = [Region(1, m1, "#3c5ac8"), Region(2, m2, "#5c80ca")]
    assert _group_is_fillable(group, float(m1.sum() + m2.sum()), img) is False


def test_stop_span_registers_cyclic_travel():
    # a cyclic stop set (returns to its starting colour) must register travel, not ~0.
    from vectormark.gradient import _stop_span, _MIN_STOP_SPAN
    cyclic = [(0.0, "#ff0000"), (0.5, "#0000ff"), (1.0, "#ff0000")]   # red -> blue -> red
    assert _stop_span(cyclic) > _MIN_STOP_SPAN
    # a monotone ramp is unchanged (max-from-first == last-from-first)
    mono = [(0.0, "#ff0000"), (0.5, "#7f007f"), (1.0, "#0000ff")]
    # this triple is ~collinear in OKLab, so max-from-first == last-from-first; for a
    # curved/cyclic stop path the new span is >= the old, never less (so the gate only
    # admits MORE fields, never fewer).
    assert abs(_stop_span(mono) - _stop_span([mono[0], mono[-1]])) < 1e-9


def test_fit_stretch_masks_outside_pixels():
    # a smooth footprint whose bbox also contains a contrasting OUTSIDE-mask patch (a hole):
    # the bright hole colour must NOT bleed into the downsampled fill (it is replaced by the
    # footprint mean before downsampling).
    from vectormark.gradient import _fit_stretch
    import base64, io
    from PIL import Image
    h, w = 64, 64
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.clip(30 + xx * 3, 0, 255)        # smooth blue->red-ish horizontal field
    img[:, :, 2] = np.clip(200 - xx * 3, 0, 255)
    mask = np.ones((h, w), bool); mask[24:40, 24:40] = False   # a hole
    img[~mask] = (0, 255, 0)                            # bright green in the hole (outside-mask)
    model = _fit_stretch(mask, img)
    assert model is not None and model["kind"] == "raster"
    small = np.asarray(Image.open(io.BytesIO(base64.b64decode(model["png_b64"]))).convert("RGB"))
    # the green hole colour must be absent from the downsampled fill (it was masked out)
    assert float(np.linalg.norm(small.reshape(-1, 3).astype(float) - np.array([0, 255, 0]), axis=1).min()) > 60.0


def test_component_fill_uses_searched_radial_for_sphere():
    # a spherical/glossy field (bright off-centre highlight fading outward): the cheap heuristic
    # settles for a poor linear; the searched parametric finds the radial. _component_fill must
    # return the radial, not a linear.
    from vectormark.gradient import _component_fill
    h, w = 80, 80
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - 28, yy - 24) / np.hypot(w, h)          # highlight off-centre (upper-left)
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((250, 40), (250, 30), (250, 50))):
        img[:, :, ch] = a + r * (b - a)
    img = img.round().astype(np.uint8)
    model = _component_fill(np.ones((h, w), bool), img)
    assert model is not None and model["kind"] == "radial"
