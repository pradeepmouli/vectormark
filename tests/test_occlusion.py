# SPDX-License-Identifier: MIT
import numpy as np
from vectormark.types import Region
from vectormark.occlusion import ScenePrimitive, region_adjacency


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
