import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.framework import Proposal, optimize
from vectormark.optimizer.gate import rasterize
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


def test_framework_rejects_multi_id_proposal_that_drops_consumed_coverage():
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

    def drop_second_object_pass(os, ms):
        return [Proposal((1, 2), [os[0]])]

    out = optimize(objs, masks, [drop_second_object_pass])

    assert [obj.id for obj in out] == [1, 2]


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


def test_framework_accepts_single_id_split_and_assigns_replacement_masks():
    obj = _rect_obj(1, 20, 10, x=0, y=0)
    masks = {1: rasterize(obj.flat, (20, 30))}

    def split_pass(os, ms):
        return [Proposal((1,), [_rect_obj(1, 10, 10, x=0, y=0), _rect_obj(9, 10, 10, x=10, y=0)])]

    def verify_split_masks(os, ms):
        assert [current.id for current in os] == [1, 9]
        expected_left = rasterize(os[0].flat, (20, 30))
        expected_right = rasterize(os[1].flat, (20, 30))
        assert np.array_equal(ms[1], expected_left)
        assert np.array_equal(ms[9], expected_right)
        assert int(ms[1].sum()) < int(masks[1].sum())
        assert int(ms[9].sum()) < int(masks[1].sum())
        return []

    out = optimize([obj], masks, [split_pass, verify_split_masks])

    assert [current.id for current in out] == [1, 9]


def test_framework_updates_region_raster_and_pass_diagnostics():
    obj = _rect_obj(1, 20, 10, x=0, y=0)
    masks = {1: rasterize(obj.flat, (20, 30))}

    def split_for_diagnostics(os, ms):
        return [Proposal((1,), [_rect_obj(1, 10, 10, x=0, y=0), _rect_obj(9, 10, 10, x=10, y=0)])]

    out = optimize([obj], masks, [split_for_diagnostics])
    by_id = {region.id: region for region in out}

    assert np.array_equal(by_id[1].raster, rasterize(by_id[1].flat, (20, 30)))
    assert np.array_equal(by_id[9].raster, rasterize(by_id[9].flat, (20, 30)))
    assert by_id[1].diagnostics["split_for_diagnostics"]["accepted"] is True
    assert by_id[1].diagnostics["split_for_diagnostics"]["proposal_ids"] == [1]
    assert by_id[9].diagnostics["split_for_diagnostics"]["proposal_ids"] == [1]


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


def test_framework_aggregate_gate_is_independent_of_replacement_order():
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

    def forward_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(8, 10, 10, x=10), _rect_obj(7, 10, 10)])]

    def reverse_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(7, 10, 10), _rect_obj(8, 10, 10, x=10)])]

    forward = optimize(objs, masks, [forward_pass])
    reverse = optimize(objs, masks, [reverse_pass])

    assert [obj.id for obj in forward] == [obj.id for obj in reverse]


def test_framework_multi_id_matching_replacement_uses_own_mask_not_union():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def replace_one_consumed_id(os, ms):
        return [Proposal((1, 2), [_path_square_obj(1, size=9.0), os[1]])]

    out = optimize(objs, masks, [replace_one_consumed_id], budget=0.25)

    assert [obj.id for obj in out] == [1, 2]
    assert out[0].exact.kind == "path"


def test_framework_tiebreaks_same_consumed_ids_by_replacement_values():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    def conflicting_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((1,), [_rect_obj(8, 10, 10)]),
        ]

    out = optimize(objs, masks, [conflicting_pass])

    assert [obj.id for obj in out] == [8]


def test_framework_rejects_duplicate_replacement_ids_without_mutation():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    original_masks = {obj_id: mask.copy() for obj_id, mask in masks.items()}

    def duplicate_id_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 20, 10), _rect_obj(9, 20, 10)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1, 2]
        for obj_id in (1, 2):
            assert np.array_equal(ms[obj_id], original_masks[obj_id])
        return []

    out = optimize(objs, masks, [duplicate_id_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_rejects_empty_replacements_without_mutation():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True
    original_mask = masks[1].copy()

    def empty_pass(os, ms):
        return [Proposal((1,), [])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1]
        assert np.array_equal(ms[1], original_mask)
        return []

    out = optimize(objs, masks, [empty_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1]


def test_framework_canonicalizes_input_order_for_later_passes():
    objs = [_rect_obj(2, 5, 5, x=5), _rect_obj(1, 5, 5)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:5, 0:5] = True
    masks[2][0:5, 5:10] = True

    def verify_canonical(os, ms):
        assert [obj.id for obj in os] == [1, 2]
        return []

    out = optimize(objs, masks, [verify_canonical])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_rejects_duplicate_consumed_ids_without_mutation():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True
    original_mask = masks[1].copy()

    def duplicate_consumed_pass(os, ms):
        return [Proposal((1, 1), [_rect_obj(9, 10, 10)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1]
        assert np.array_equal(ms[1], original_mask)
        return []

    out = optimize(objs, masks, [duplicate_consumed_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1]


def test_framework_rejects_same_pass_consumption_of_created_id():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    def chaining_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((9,), [_rect_obj(10, 10, 10)]),
        ]

    def verify_only_first_proposal_applied(os, ms):
        assert [obj.id for obj in os] == [9]
        assert 9 in ms
        assert 10 not in ms
        return []

    out = optimize(objs, masks, [chaining_pass, verify_only_first_proposal_applied])

    assert [obj.id for obj in out] == [9]


def test_framework_rejects_same_pass_reuse_of_created_replacement_id():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def duplicate_created_id_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((2,), [_rect_obj(9, 10, 10, x=10)]),
        ]

    out = optimize(objs, masks, [duplicate_created_id_pass])

    assert [obj.id for obj in out] == [2, 9]
    assert len({obj.id for obj in out}) == len(out)
