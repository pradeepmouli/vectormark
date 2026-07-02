import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill, LinearGradientFill
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.vector_region import VectorRegion
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
) -> VectorRegion:
    return VectorRegion(obj_id, _path_from_poly(poly), fill, z, poly)


def _rect(
    obj_id: int,
    *,
    bounds: tuple[float, float, float, float],
    fill: object = FlatFill("#112233"),
) -> VectorRegion:
    minx, miny, maxx, maxy = bounds
    poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    return _obj(obj_id, poly, fill=fill)


def test_symmetry_pass_replaces_mirror_pair_with_use():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    objects = [right, left]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = optimize(objects, masks, [symmetry_pass])

    assert proposals[0].obj_ids == (1, 2)
    assert [obj.id for obj in proposals[0].new_objects] == [1, 2]
    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"
    assert out[1].current.kind == "use"
    assert out[1].current.params == {
        "href_obj_id": 1,
        "transform": (-1.0, 0.0, 0.0, 1.0, 80.0, 0.0),
        "fill": "#abcdef",
    }


def test_symmetry_pass_reconstructs_self_symmetric_object_as_exact_path():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = Polygon(
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(1, symmetric_contour)
    masks = {obj.id: _mask_for_polygon(obj.footprint)}

    out = optimize([obj], masks, [symmetry_pass])

    assert len(out) == 2
    assert [current.current.kind for current in out] == ["path", "use"]
    assert out[0].id == 1
    assert out[0].current.kind == "path"
    assert out[0].current != obj.current
    assert out[1].current.params["href_obj_id"] == 1
    reflected = out[0].footprint.union(out[1].footprint).symmetric_difference(symmetric_contour)
    assert reflected.area < 1e-6


def test_symmetry_pass_preserves_self_symmetric_native_primitive():
    poly = Polygon([(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)])
    obj = VectorRegion(
        1,
        Shape("rect", {"x": 20.0, "y": 20.0, "w": 40.0, "h": 40.0}),
        FlatFill("#112233"),
        0,
        poly,
    )

    assert symmetry_pass([obj], {obj.id: _mask_for_polygon(poly)}) == []


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
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = optimize(objects, masks, [symmetry_pass])

    assert all(obj.current.kind != "use" for proposal in proposals for obj in proposal.new_objects)
    assert [obj.current.kind for obj in out] == ["path", "path"]


def test_symmetry_pass_is_deterministic_for_unordered_inputs():
    left = _rect(3, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(9, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in [left, right]}

    first = symmetry_pass([right, left], masks)
    second = symmetry_pass([left, right], masks)

    assert first == second
    assert first[0].obj_ids == (3, 9)
    assert first[0].new_objects[1].current.params["href_obj_id"] == 3


def test_symmetry_pass_rejects_pair_when_mask_gate_fails():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    masks = {
        left.id: _mask_for_polygon(left.footprint),
        right.id: _mask_for_polygon(Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])),
    }

    proposals = symmetry_pass([left, right], masks)
    out = optimize([left, right], masks, [symmetry_pass])

    assert all(2 not in proposal.obj_ids for proposal in proposals)
    assert [obj.current.kind for obj in out] == ["path", "path"]
