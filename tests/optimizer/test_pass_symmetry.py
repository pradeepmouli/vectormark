import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill, LinearGradientFill
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.optobject import OptObject
from vectormark.optimizer.passes.symmetry import symmetry_pass
from vectormark.fit import Shape


def _mask_for_polygon(poly: Polygon, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(poly, shape_hw)


def _path_from_poly(poly: Polygon) -> Shape:
    coords = list(poly.exterior.coords)
    body = " ".join(f"L{x} {y}" for x, y in coords[1:])
    return Shape("path", {"d": f"M{coords[0][0]} {coords[0][1]} {body} Z"})


def _obj(
    obj_id: int,
    poly: Polygon,
    *,
    fill: object = FlatFill("#112233"),
    z: int = 0,
) -> OptObject:
    return OptObject(obj_id, _path_from_poly(poly), fill, z, poly)


def _rect(
    obj_id: int,
    *,
    bounds: tuple[float, float, float, float],
    fill: object = FlatFill("#112233"),
) -> OptObject:
    minx, miny, maxx, maxy = bounds
    poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    return _obj(obj_id, poly, fill=fill)


def test_symmetry_pass_replaces_mirror_pair_with_use():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    objects = [right, left]
    masks = {obj.id: _mask_for_polygon(obj.flat) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = optimize(objects, masks, [symmetry_pass])

    assert proposals[0].obj_ids == (1, 2)
    assert [obj.id for obj in proposals[0].new_objects] == [1, 2]
    assert [obj.id for obj in out] == [1, 2]
    assert out[0].exact.kind == "path"
    assert out[1].exact.kind == "use"
    assert out[1].exact.params == {
        "href_obj_id": 1,
        "transform": (-1.0, 0.0, 0.0, 1.0, 80.0, 0.0),
        "fill": "#abcdef",
    }


def test_symmetry_pass_reconstructs_self_symmetric_object_as_exact_path():
    diamond = Polygon([(40.0, 16.0), (58.0, 32.0), (40.0, 48.0), (22.0, 32.0)])
    obj = _obj(1, diamond)
    masks = {obj.id: _mask_for_polygon(obj.flat)}

    out = optimize([obj], masks, [symmetry_pass])

    assert len(out) == 1
    assert out[0].id == 1
    assert out[0].exact.kind == "path"
    assert out[0].exact != obj.exact
    reflected = out[0].flat.symmetric_difference(
        Polygon([(40.0, 16.0), (58.0, 32.0), (40.0, 48.0), (22.0, 32.0)])
    )
    assert reflected.area < 1e-6


def test_symmetry_pass_skips_non_flat_fill_pair_use():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(
        2,
        bounds=(58.0, 20.0, 70.0, 34.0),
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )
    objects = [left, right]
    masks = {obj.id: _mask_for_polygon(obj.flat) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = optimize(objects, masks, [symmetry_pass])

    assert all(obj.exact.kind != "use" for proposal in proposals for obj in proposal.new_objects)
    assert [obj.exact.kind for obj in out] == ["path", "path"]


def test_symmetry_pass_is_deterministic_for_unordered_inputs():
    left = _rect(3, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(9, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    masks = {obj.id: _mask_for_polygon(obj.flat) for obj in [left, right]}

    first = symmetry_pass([right, left], masks)
    second = symmetry_pass([left, right], masks)

    assert first == second
    assert first[0].obj_ids == (3, 9)
    assert first[0].new_objects[1].exact.params["href_obj_id"] == 3


def test_symmetry_pass_rejects_pair_when_mask_gate_fails():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    masks = {
        left.id: _mask_for_polygon(left.flat),
        right.id: _mask_for_polygon(Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])),
    }

    proposals = symmetry_pass([left, right], masks)
    out = optimize([left, right], masks, [symmetry_pass])

    assert all(2 not in proposal.obj_ids for proposal in proposals)
    assert [obj.exact.kind for obj in out] == ["path", "path"]
