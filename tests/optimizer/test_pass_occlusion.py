import numpy as np

from vectormark.optimizer.framework import optimize
from vectormark.optimizer.passes.occlusion import occlusion_pass
from vectormark.optimizer.passes.primitives import primitives_pass
from vectormark.optimizer.trace import trace_regions
from vectormark.pipeline import Options


def _synthetic_mastercard(h=240, w=360, gap=58, r=104):
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((h, w, 3), 255, np.uint8)
    img[left] = (235, 0, 27)
    img[right] = (247, 158, 27)
    img[left & right] = (255, 95, 0)
    return img


def test_occlusion_pass_replaces_adjacent_traces_with_scene_branch():
    regions, masks = trace_regions(_synthetic_mastercard(), Options())

    optimized = optimize(regions, masks, [primitives_pass, occlusion_pass])

    assert len(optimized) == 1
    branch = optimized[0]
    assert branch.is_branch
    assert branch.current is None
    assert branch.diagnostics["occlusion"]["accepted"] is True
    assert branch.diagnostics["occlusion"]["children"] == 3
    assert [leaf.current.kind for leaf in branch.leaves()] == ["circle", "circle", "path"]
