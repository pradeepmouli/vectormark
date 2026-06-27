import numpy as np
from vectormark.surface_merge import seam_is_soft, gradients_continuous, merge_surfaces
from vectormark.fill_fit import fit_fill
from vectormark.fill_fit import fit_fill as _ff
from vectormark.types import Region
from vectormark.candidate import LinearGradientFill, RadialGradientFill

_AnyGradient = (LinearGradientFill, RadialGradientFill)


def _ramp(h, w, c0, c1):
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / (w - 1)
        rgb[:, x] = [round(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]
    return rgb


def _full_ramp_split():
    # one blue->red ramp, split into left and right halves as two masks over one canvas
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    return rgb, a, b


def test_within_ramp_seam_is_soft():
    rgb, a, b = _full_ramp_split()
    assert seam_is_soft(a, b, rgb)                       # one continuous ramp sliced in two


def test_hard_color_step_is_not_soft():
    # left half ramps to red; right half is a uniform blue patch (a "dot") -> sharp step
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    assert not seam_is_soft(a, b, rgb)


# NOTE: a same-color-at-seam feature is not distinguishable by seam_is_soft — when a
# feature's color matches the ramp at the shared border there is genuinely no step, so
# seam_is_soft correctly reads soft. The same-color protection lives at the merge level
# (the union-fits-gradient guard in Task 3), tested in test_surface_merge merge tests.


def test_non_adjacent_masks_are_not_soft():
    rgb, a, _b = _full_ramp_split()
    far = np.zeros_like(a); far[:5, 70:75] = True         # disjoint from a (a is left half)
    assert not seam_is_soft(a, far, rgb)


# ── B: gradient-continuity path (both regions wide enough to fit a gradient) ──

def test_two_gradient_halves_are_continuous():
    # each half of a wide ramp spans enough colour to fit a gradient -> compare at the seam
    rgb, a, b = _full_ramp_split()
    fa, fb = fit_fill(a, rgb, flat_hex="#000000"), fit_fill(b, rgb, flat_hex="#000000")
    assert gradients_continuous(fa, a, fb, b)


def test_gradient_meets_flat_is_not_continuous():
    # one side ramps, the other is a uniform patch (flat) -> B does not apply (False)
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    fa, fb = fit_fill(a, rgb, flat_hex="#000000"), fit_fill(b, rgb, flat_hex="#143CD2")
    assert not gradients_continuous(fa, a, fb, b)


# ── Task 3: merge_surfaces ────────────────────────────────────────────────────

def _region(label, mask, hex_):
    return Region(label=label, mask=mask, color_hex=hex_)


def test_ramp_halves_merge_into_one_gradient():
    # wide halves each fit a gradient -> B path
    # NOTE: the brief says "a horizontal ramp may fit RADIAL — that's fine; the tests
    # assert 'is a gradient'"; we check _AnyGradient accordingly.
    rgb, a, b = _full_ramp_split()
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8")),
              (_region(1, b, "#DC2814"), _ff(b, rgb, flat_hex="#DC2814"))]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1
    region, fill = out[0]
    assert isinstance(fill, _AnyGradient)
    assert region.mask.sum() == (a | b).sum()           # union silhouette, clean


def test_narrow_bands_collapse_to_one_gradient():
    # narrow strips of one smooth ramp collapse to a single gradient: a strip too narrow
    # to fit a gradient devolves to flat and merges via the A path; a wide-enough one via
    # B. Either way the outcome is one gradient.
    H, W = 40, 96
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    filled = []
    for k in range(0, W, 8):
        m = np.zeros((H, W), bool); m[:, k:k + 8] = True
        filled.append((_region(k, m, "#808080"), _ff(m, rgb, flat_hex="#808080")))
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1 and isinstance(out[0][1], _AnyGradient)


def test_flat_dot_on_ramp_stays_separate():
    # the "dot" has a HARD border against the ramp -> hard seam -> never merged,
    # even though its blue is in the wing's color family.
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8")),
              (_region(1, b, "#143CD2"), _ff(b, rgb, flat_hex="#143CD2"))]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 2                                 # NOT merged

def test_two_distinct_flats_do_not_merge():
    # two solid colors meeting at a soft-ish AA edge: union is NOT a gradient -> no merge.
    H, W = 40, 80
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = (200, 40, 40); rgb[:, W // 2:] = (40, 160, 60)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#C82828"), _ff(a, rgb, flat_hex="#C82828")),
              (_region(1, b, "#28A03C"), _ff(b, rgb, flat_hex="#28A03C"))]
    assert len(merge_surfaces(filled, rgb)) == 2
