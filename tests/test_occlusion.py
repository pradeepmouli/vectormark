# SPDX-License-Identifier: MIT
import numpy as np
from vectormark.types import Region
from vectormark.occlusion import ScenePrimitive, region_adjacency, has_bite


def _region(label, mask, color="#000000"):
    return Region(label, mask.astype(bool), color)


def test_scene_primitive_holds_geometry():
    p = ScenePrimitive(kind="circle", params={"cx": 1.0, "cy": 2.0, "r": 3.0}, color_hex="#FF0000", z=0)
    assert p.kind == "circle" and p.params["r"] == 3.0 and p.color_hex == "#FF0000" and p.z == 0


def test_region_adjacency_touching_vs_separate():
    a = np.zeros((20, 30), bool); a[5:15, 4:14] = True
    b = np.zeros((20, 30), bool); b[5:15, 14:24] = True   # shares the x=14 seam with a
    c = np.zeros((20, 30), bool); c[5:15, 26:29] = True   # gap from b
    adj = region_adjacency([_region(1, a), _region(2, b), _region(3, c)])
    assert adj[1] == {2} and adj[2] == {1} and adj[3] == set()


def test_has_bite_crescent_vs_convex():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 35) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    occ = (xx - 60) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    crescent = disk & ~occ
    assert has_bite(crescent) is True
    assert has_bite(disk) is False                      # convex
    rect = np.zeros((H, W), bool); rect[10:70, 10:40] = True
    assert has_bite(rect) is False


from vectormark.occlusion import label_boundary, complete_primitive


def test_label_boundary_marks_the_bite_as_seam():
    H = W = 90
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 38) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    occ = (xx - 66) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    crescent = disk & ~occ
    other = occ & ~disk                                  # the occluding region's visible part
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    assert contour.shape[1] == 2 and seam.dtype == bool and len(seam) == len(contour)
    seam_x = contour[seam][:, 0].mean()
    own_x = contour[~seam][:, 0].mean()
    assert seam_x > own_x                                # the bite is on the +x side
    assert seam.any() and (~seam).any()


def test_complete_primitive_recovers_full_disk_from_crescent():
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cx0, cy0, r0 = 45.0, 60.0, 34.0
    disk = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r0 ** 2
    occ = (xx - 78) ** 2 + (yy - 60) ** 2 <= 34 ** 2
    crescent = disk & ~occ
    other = occ & ~disk
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    prim = complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0)
    assert prim is not None and prim["kind"] == "circle"
    assert abs(prim["params"]["cx"] - cx0) < 2.0
    assert abs(prim["params"]["cy"] - cy0) < 2.0
    assert abs(prim["params"]["r"] - r0) < 2.0


def test_complete_primitive_rejects_when_own_arc_too_short():
    contour = np.array([[0, 0], [1, 0], [2, 1], [2, 2], [1, 3], [0, 3]], float)
    seam = np.ones(len(contour), bool); seam[0] = False
    assert complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0) is None


import re
from vectormark.occlusion import intersection_lens_d


def test_intersection_lens_path_two_arcs_between_crossing_points():
    a = {"cx": 0.0, "cy": 0.0, "r": 6.0}
    b = {"cx": 8.0, "cy": 0.0, "r": 6.0}
    d = intersection_lens_d(a, b)
    assert d is not None
    assert d.count("A") == 2 and d.startswith("M") and d.rstrip().endswith("Z")
    nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
    assert any(abs(n - 4.0) < 0.5 for n in nums)          # crossing points at x=4 (midpoint)


def test_intersection_lens_none_when_disjoint():
    a = {"cx": 0.0, "cy": 0.0, "r": 3.0}
    b = {"cx": 20.0, "cy": 0.0, "r": 3.0}
    assert intersection_lens_d(a, b) is None


from vectormark.occlusion import primitive_mask, stack_agreement


def test_primitive_mask_circle():
    m = primitive_mask({"kind": "circle", "params": {"cx": 10.0, "cy": 10.0, "r": 5.0}}, 20, 20)
    assert m[10, 10] and not m[10, 17] and m.dtype == bool


def test_stack_agreement_high_for_true_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    db = (xx - 62) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    red = da & ~db; yellow = db & ~da; lens = da & db
    regions = [_region(1, red, "#FF0000"), _region(2, yellow, "#FFFF00"), _region(3, lens, "#FFA500")]
    prims = [
        {"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 30.0}, "color": "#FF0000", "z": 0},
        {"kind": "circle", "params": {"cx": 62.0, "cy": 50.0, "r": 30.0}, "color": "#FFFF00", "z": 1},
    ]
    lens_shape = {"mask_color": "#FFA500", "lens_of": (0, 1)}
    assert stack_agreement(prims, lens_shape, regions, H, W) > 0.97


def test_stack_agreement_low_for_wrong_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    regions = [_region(1, da, "#FF0000")]
    prims = [{"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 45.0}, "color": "#FF0000", "z": 0}]
    assert stack_agreement(prims, None, regions, H, W) < 0.9   # oversized disk disagrees


from vectormark.types import Axis
from vectormark.fit import Shape
from vectormark.occlusion import reconstruct_scene


def _two_disk_mark(H=140, W=200, gap=24, r=44):
    yy, xx = np.ogrid[:H, :W]
    cx = W // 2
    da = (xx - (cx - gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    db = (xx - (cx + gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    return da, db, cx


def test_reconstruct_scene_two_disks_with_lens():
    H, W = 140, 200
    da, db, cx = _two_disk_mark(H, W)
    regions = [
        _region(1, da & ~db, "#FF0000"),
        _region(2, db & ~da, "#FFFF00"),
        _region(3, da & db, "#FFA500"),
    ]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=float(cx)), (H, W))
    prims = [e for e in reconstructed if isinstance(e, ScenePrimitive)]
    lenses = [e for e in reconstructed if isinstance(e, Shape)]
    assert len(prims) == 2 and len(lenses) == 1
    assert abs(prims[0].params["r"] - prims[1].params["r"]) < 1.0
    assert abs((prims[0].params["cx"] + prims[1].params["cx"]) / 2 - cx) < 1.0
    assert remaining == []


def test_reconstruct_scene_passes_through_non_occluded():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    band = np.zeros((H, W), bool); band[20:40, 10:70] = True     # convex, no bite
    regions = [_region(1, band, "#062336")]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=40.0), (H, W))
    assert reconstructed == [] and [r.label for r in remaining] == [1]
