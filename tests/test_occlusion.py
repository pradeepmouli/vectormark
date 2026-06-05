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


from vectormark.occlusion import label_boundary


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
