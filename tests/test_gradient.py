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


def test_ramp_groups_groups_a_monotonic_ramp():
    from vectormark.gradient import _ramp_groups
    # 4 adjacent bands stepping blue->magenta (a clear OKLab ramp)
    regions = _hstrip_regions(["#2563eb", "#7b3fc4", "#b13a9e", "#db2777"])
    groups = _ramp_groups(regions)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_ramp_groups_rejects_flat_and_too_few():
    from vectormark.gradient import _ramp_groups
    flat = _hstrip_regions(["#2563eb", "#2563eb", "#2563eb"])   # no variation
    assert _ramp_groups(flat) == []
    two = _hstrip_regions(["#2563eb", "#db2777"])               # only 2 -> not a gradient
    assert _ramp_groups(two) == []


def test_ramp_groups_rejects_nonramp_colors():
    from vectormark.gradient import _ramp_groups
    # adjacent but colors are not collinear in OKLab (zig-zag hues)
    regions = _hstrip_regions(["#ff0000", "#00ff00", "#0000ff", "#00ff00"])
    assert _ramp_groups(regions) == []


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
    # left half: a 4-band blue->magenta linear ramp; right: one flat green block
    img = _linear_gradient_image(h, 80, (0, 30), (79, 30),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    full = np.zeros((h, w, 3), np.uint8)
    full[:, :80] = img
    full[:, 80:] = (20, 160, 60)
    # build the quantized regions the way the pipeline would (4 ramp bands + 1 flat)
    regions = []
    for i in range(4):
        m = np.zeros((h, w), bool); m[:, i * 20:(i + 1) * 20] = True
        regions.append(Region(label=i + 1, mask=m,
                              color_hex="#%02x%02x%02x" % tuple(np.median(full[m], axis=0).astype(int))))
    gm = np.zeros((h, w), bool); gm[:, 80:] = True
    regions.append(Region(label=5, mask=gm, color_hex="#149c3c"))
    fills, remaining = detect_gradients(regions, full)
    assert len(fills) == 1                               # the ramp became one gradient fill
    assert fills[0][1]["kind"] == "linear"
    assert {r.label for r in remaining} == {5}           # the flat green block remains


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


def test_detect_gradients_dissolves_unfittable_group_back_to_flats():
    from vectormark.color import oklab_to_srgb, srgb_to_oklab
    from vectormark.gradient import _ramp_groups, detect_gradients
    from vectormark.types import Region
    # Six flat bands whose colours are collinear AND have distinct projections in OKLab,
    # so _ramp_groups accepts them (group IS found). But the SPATIAL layout is a
    # high-frequency zig-zag along the colour line (offsets 0, 1, .2, .8, .4, .6), so no
    # single linear OR radial model reproduces the solid blocks within _GATE_DELTA_E
    # (empirically dE_lin~0.117, dE_rad~0.098 >> 0.05) -> fit_gradient returns None and
    # detect_gradients dissolves the bands back into `remaining` via its `continue`.
    l0 = srgb_to_oklab(np.array([20, 50, 250])[None] / 255.0)[0]
    l1 = srgb_to_oklab(np.array([250, 20, 90])[None] / 255.0)[0]

    def hex_at(o):
        rgb = (np.clip(oklab_to_srgb((l0 + o * (l1 - l0))[None])[0], 0, 1) * 255).round().astype(int)
        return "#%02x%02x%02x" % tuple(rgb)

    spatial = [0.0, 1.0, 0.2, 0.8, 0.4, 0.6]               # zig-zag, not monotone in space
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

    assert len(_ramp_groups(regions)) == 1                 # the group IS found
    fills, remaining = detect_gradients(regions, img)
    assert fills == []                                     # but no model fits -> rejected
    assert {r.label for r in remaining} == {1, 2, 3, 4, 5, 6}  # all bands fall back to flats


def test_dominant_blob_fraction():
    from vectormark.gradient import _dominant_blob_fraction
    m = np.zeros((20, 40), bool)
    m[2:18, 2:18] = True                       # one 16x16 blob, rest empty
    assert _dominant_blob_fraction(m) == 1.0
    m[2:18, 22:38] = True                      # add a second, equal, disconnected blob
    assert abs(_dominant_blob_fraction(m) - 0.5) < 1e-9
    assert _dominant_blob_fraction(np.zeros((5, 5), bool)) == 0.0   # empty -> 0
