import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.passes.straighten import straighten_pass
from vectormark.optimizer.vector_region import VectorRegion


def test_straighten_pass_replaces_near_collinear_quadratic_with_line():
    shape = Shape("path", {"d": "M0 0 Q0.2 5 0 10 L10 10 L10 0 Z"})
    region = VectorRegion(1, shape, FlatFill("#123456"), 0)

    proposals = straighten_pass(
        [region],
        {region.id: rasterize(region.footprint, (16, 16))},
        epsilon=0.5,
    )

    assert len(proposals) == 1
    current = proposals[0].new_objects[0].current
    assert current is not None
    assert "Q" not in current.params["d"]


def test_straighten_pass_skips_gradient_fills():
    from vectormark.candidate import LinearGradientFill

    shape = Shape("path", {"d": "M0 0 Q0.2 5 0 10 L10 10 L10 0 Z"})
    region = VectorRegion(
        1,
        shape,
        LinearGradientFill({"x1": 0, "y1": 0, "x2": 10, "y2": 0}, [(0, "#000000"), (1, "#ffffff")]),
        0,
    )

    assert straighten_pass([region], {region.id: rasterize(region.footprint, (16, 16))}, epsilon=0.5) == []
