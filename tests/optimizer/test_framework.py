import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import Proposal, optimize
from vectormark.optimizer.optobject import OptObject


def _rect_obj(i, w, h):
    return OptObject(
        i,
        Shape("rect", {"x": 0.0, "y": 0.0, "w": float(w), "h": float(h)}),
        FlatFill("#000"),
        0,
    )


def test_framework_accepts_good_rejects_bad():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    good = lambda os, ms: [Proposal((1,), [os[0]])]
    out = optimize(objs, masks, [good])
    assert len(out) == 1

    bad = lambda os, ms: [Proposal((1,), [_rect_obj(1, 2, 2)])]
    out2 = optimize(objs, masks, [bad])
    assert abs(out2[0].exact.params["w"] - 10) < 1e-9


def test_framework_orders_multi_id_proposals_and_unions_masks_for_new_ids():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 6, 6)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][10:16, 10:16] = True

    def first_pass(os, ms):
        return [
            Proposal((2,), [_rect_obj(2, 6, 6)]),
            Proposal((1, 2), [_rect_obj(1, 10, 10), _rect_obj(9, 10, 10)]),
        ]

    def second_pass(os, ms):
        assert [obj.id for obj in os] == [1, 9]
        assert np.array_equal(ms[1], masks[1])
        assert np.array_equal(ms[9], masks[1] | masks[2])
        return []

    out = optimize(objs, masks, [first_pass, second_pass])
    assert [obj.id for obj in out] == [1, 9]
