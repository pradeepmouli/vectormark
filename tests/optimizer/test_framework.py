import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import Proposal, optimize
from vectormark.optimizer.optobject import OptObject


def _rect_obj(i, w, h, *, x=0.0, y=0.0):
    return OptObject(
        i,
        Shape("rect", {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}),
        FlatFill("#000"),
        0,
    )


def _path_square_obj(i, *, x=0.0, y=0.0, size=10.0):
    x2 = x + size
    y2 = y + size
    d = f"M{x} {y} L{x2} {y} L{x2} {y2} L{x} {y2} Z"
    return OptObject(i, Shape("path", {"d": d}), FlatFill("#000"), 0)


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
    masks[2][0:10, 10:16] = True

    def first_pass(os, ms):
        return [
            Proposal((2,), [_rect_obj(2, 6, 6)]),
            Proposal((1, 2), [_rect_obj(9, 15, 9)]),
        ]

    def second_pass(os, ms):
        assert [obj.id for obj in os] == [9]
        assert np.array_equal(ms[9], masks[1] | masks[2])
        return []

    out = optimize(objs, masks, [first_pass, second_pass])
    assert [obj.id for obj in out] == [9]


def test_framework_rejects_multi_id_new_id_without_union_coverage():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def bad_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 2, 2, x=0, y=0)])]

    out = optimize(objs, masks, [bad_pass])

    assert [obj.id for obj in out] == [1, 2]
    assert [obj.exact.params["w"] for obj in out] == [10.0, 10.0]


def test_framework_accepts_multi_id_new_id_with_union_coverage():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    union_mask = masks[1] | masks[2]

    def merge_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 19, 9, x=0, y=0)])]

    def verify_union_mask(os, ms):
        assert [obj.id for obj in os] == [9]
        assert np.array_equal(ms[9], union_mask)
        return []

    out = optimize(objs, masks, [merge_pass, verify_union_mask])

    assert [obj.id for obj in out] == [9]
    assert out[0].exact.params["w"] == 19.0


def test_framework_rejects_replacement_id_aliasing_live_object():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
        _rect_obj(3, 5, 5, x=0, y=10),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
        3: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    masks[3][10:15, 0:5] = True
    original_masks = {obj_id: mask.copy() for obj_id, mask in masks.items()}

    def bad_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(3, 20, 10, x=0, y=0)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1, 2, 3]
        for obj_id in (1, 2, 3):
            assert np.array_equal(ms[obj_id], original_masks[obj_id])
        return []

    out = optimize(objs, masks, [bad_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1, 2, 3]
    for obj_id in (1, 2, 3):
        assert np.array_equal(masks[obj_id], original_masks[obj_id])


def test_framework_sorts_replacements_before_later_passes():
    objs = [_rect_obj(1, 5, 5), _rect_obj(2, 5, 5, x=5)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:5, 0:5] = True
    masks[2][0:5, 5:10] = True

    def reverse_order_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(8, 9, 4), _rect_obj(7, 9, 4)])]

    def verify_sorted(os, ms):
        assert [obj.id for obj in os] == [7, 8]
        return []

    out = optimize(objs, masks, [reverse_order_pass, verify_sorted], budget=2.0)

    assert [obj.id for obj in out] == [7, 8]


def test_framework_multi_id_matching_replacement_uses_own_mask_not_union():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def replace_one_consumed_id(os, ms):
        return [Proposal((1, 2), [_path_square_obj(1, size=9.0)])]

    out = optimize(objs, masks, [replace_one_consumed_id], budget=0.25)

    assert [obj.id for obj in out] == [1]
    assert out[0].exact.kind == "path"
