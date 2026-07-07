import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill, LinearGradientFill
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.vector_region import VectorRegion, to_polygon
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


def test_symmetry_pass_replaces_mirror_pair_with_baked_path():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    objects = [right, left]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = optimize(objects, masks, [symmetry_pass])

    assert proposals[0].obj_ids == (1, 2)
    assert [obj.id for obj in proposals[0].new_objects] == [1, 2]
    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"
    assert out[1].current.kind == "path"
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].diagnostics["symmetry"]["matched_source"] == 1


def test_symmetry_pass_reconstructs_self_symmetric_object_as_single_region():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = Polygon(
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(1, symmetric_contour)
    masks = {obj.id: _mask_for_polygon(obj.footprint)}

    out = optimize([obj], masks, [symmetry_pass])

    assert len(out) == 1
    region = out[0]
    assert region.id == 1
    assert region.is_branch
    assert len(region.children) == 2
    assert region.children[0].current.kind == "path"
    assert region.children[1].current.kind == "use"
    assert region.children[1].current.params["href_obj_id"] == region.children[0].id
    assert region.diagnostics["symmetry"]["mode"] == "self"


def test_symmetry_pass_forwards_cubic_path_option_to_self_fit():
    shape = Shape(
        "path",
        {
            "d": (
                "M390 302.5 L256 302.5 "
                "Q245.79 281.27 240.5 258 "
                "Q241.25 257.25 242 256.5 "
                "L404 256.5 "
                "Q404.75 257.25 405.5 258 "
                "Q400.2 281.26 390 302.5 Z"
            )
        },
    )
    obj = VectorRegion(1, shape, FlatFill("#fd8e27"), 0, to_polygon(shape))

    out = optimize(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (340, 460))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_branch
    assert "C" in str(out[0].children[0].current.params["d"])


def test_symmetry_pass_uses_fit_path_when_trapezoid_fitter_is_unavailable(monkeypatch):
    import vectormark.optimizer.passes.symmetry as symmetry_module

    monkeypatch.setattr(symmetry_module, "rounded_trapezoid_half_fit", lambda *args, **kwargs: None)
    shape = Shape(
        "path",
        {
            "d": (
                "M322.92 256.5 "
                "L400.35 256.5 "
                "C403.61 256.5 405.46 259.02 404.49 262.13 "
                "L393.68 296.87 "
                "C392.71 299.98 389.29 302.5 386.03 302.5 "
                "L322.92 302.5 "
                "L259.81 302.5 "
                "C256.55 302.5 253.13 299.98 252.16 296.87 "
                "L241.34 262.13 "
                "C240.38 259.02 242.23 256.5 245.49 256.5 "
                "L322.92 256.5 Z"
            )
        },
    )
    obj = VectorRegion(1, shape, FlatFill("#fd8e27"), 0, to_polygon(shape))

    out = optimize(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (340, 460))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_branch
    source = out[0].children[0]
    minx, miny, maxx, maxy = source.footprint.bounds
    assert maxx - minx > 70.0
    assert maxy - miny > 40.0
    assert source.current.params["d"].count("L") <= 3


def test_symmetry_pass_smooths_daikonic_like_cap_instead_of_preserving_facets():
    shape = Shape(
        "path",
        {
            "d": (
                "M323.08 118.6 "
                "C314.99 119.45 306.15 120.31 297.31 121.17 "
                "C288.54 123.94 279.77 126.72 271 129.5 "
                "L253.5 144 L242.5 162 L237.5 179 "
                "Q237.75 181.25 238 183.5 "
                "L323.08 183.5 L408.17 183.5 "
                "Q408.42 181.25 408.67 179 "
                "L403.67 162 L392.67 144 L375.17 129.5 "
                "C366.4 126.72 357.63 123.94 348.86 121.17 "
                "C340.02 120.31 331.18 119.45 323.08 118.6 Z"
            )
        },
    )
    obj = VectorRegion(1, shape, FlatFill("#062337"), 0, to_polygon(shape))

    out = optimize(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (240, 480))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_branch
    d = str(out[0].children[0].current.params["d"])
    assert d.count("C") == 1
    assert d.count("L") == 1
    assert "Q" not in d


def test_symmetry_pass_preserves_flat_top_width_on_daikonic_like_tip():
    shape = Shape(
        "path",
        {
            "d": (
                "M322.92 314.5 "
                "L384.5 315 "
                "Q367.72 352.4 333 379.5 "
                "Q328.43 382.04 322.92 382.5 "
                "Q317.41 382.04 312.84 379.5 "
                "Q278.12 352.4 261.34 315 "
                "L322.92 314.5 Z"
            )
        },
    )
    obj = VectorRegion(1, shape, FlatFill("#f23325"), 0, to_polygon(shape))

    out = optimize(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (420, 420))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_branch
    source = out[0].children[0]
    minx, _miny, maxx, _maxy = source.footprint.bounds
    d = str(source.current.params["d"])
    assert maxx - minx > 50.0
    assert "L" in d
    assert "C" not in d
    assert d.count("Q") <= 2


def test_symmetry_pass_reconstructs_gradient_self_symmetry_as_internal_branch():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = Polygon(
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(
        1,
        symmetric_contour,
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 96.0, "y2": 96.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )

    out = optimize([obj], {obj.id: _mask_for_polygon(obj.footprint)}, [symmetry_pass])

    assert len(out) == 1
    assert out[0].is_branch
    assert [child.current.kind for child in out[0].children] == ["path", "use"]
    assert out[0].diagnostics["symmetry"]["mode"] == "self"
    assert out[0].diagnostics["symmetry"]["accepted"] is True


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


def test_symmetry_pass_records_simple_self_symmetric_path_without_rewriting():
    poly = Polygon([(20.0, 60.0), (40.0, 20.0), (60.0, 60.0)])
    obj = _obj(1, poly)

    out = optimize([obj], {obj.id: _mask_for_polygon(poly)}, [symmetry_pass])

    assert len(out) == 1
    assert out[0].is_branch
    assert [child.current.kind for child in out[0].children] == ["path", "use"]
    assert out[0].diagnostics["symmetry"]["mode"] == "self"
    assert out[0].diagnostics["symmetry"]["accepted"] is True


def test_symmetry_pass_matches_non_flat_fill_pair_by_geometry():
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

    out = optimize(objects, masks, [symmetry_pass])

    assert [obj.current.kind for obj in out] == ["path", "path"]
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].fill == right.fill


def test_symmetry_pass_matches_recolored_pair_by_geometry():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    objects = [left, right]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = optimize(objects, masks, [symmetry_pass])

    assert [obj.current.kind for obj in out] == ["path", "path"]
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].fill == right.fill


def test_symmetry_pass_is_deterministic_for_unordered_inputs():
    left = _rect(3, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(9, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in [left, right]}

    first = symmetry_pass([right, left], masks)
    second = symmetry_pass([left, right], masks)

    assert first == second
    assert first[0].obj_ids == (3, 9)
    assert first[0].new_objects[1].current.kind == "path"
    assert first[0].new_objects[1].footprint.symmetric_difference(right.footprint).area < 1e-6


def test_symmetry_pass_does_not_use_raster_mask_as_acceptance_gate():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    masks = {
        left.id: _mask_for_polygon(left.footprint),
        right.id: _mask_for_polygon(Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])),
    }

    proposals = symmetry_pass([left, right], masks)
    out = optimize([left, right], masks, [symmetry_pass])

    assert any(2 in proposal.obj_ids for proposal in proposals)
    assert [obj.current.kind for obj in out] == ["path", "path"]
