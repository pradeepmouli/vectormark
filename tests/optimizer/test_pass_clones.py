import math

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon

from vectormark.candidate import FlatFill, LinearGradientFill
from vectormark.emit import optimizer_objects_to_svg
from vectormark.fit import Shape
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.optimizer.passes.clones import clones_pass


def _mask_for_polygon(poly: Polygon, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    from vectormark.optimizer.gate import rasterize

    return rasterize(poly, shape_hw)


def _square(
    obj_id: int,
    *,
    size: float = 12.0,
    center: tuple[float, float] = (20.0, 20.0),
    angle_deg: float = 0.0,
    fill: object = FlatFill("#112233"),
) -> VectorRegion:
    cx, cy = center
    half = size / 2.0
    poly = Polygon(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )
    if angle_deg:
        poly = affinity.rotate(poly, angle_deg, origin=(cx, cy))

    return VectorRegion(
        id=obj_id,
        current=Shape(
            "path",
            {"d": f"M{cx - half} {cy - half} L{cx + half} {cy - half} L{cx + half} {cy + half} L{cx - half} {cy + half} Z"},
        ),
        fill=fill,
        z=0,
        footprint=poly,
    )


def _circle(obj_id: int, *, center=(20.0, 20.0), radius=6.0, fill="#778899") -> VectorRegion:
    poly = Point(*center).buffer(radius, quad_segs=32)
    return VectorRegion(
        id=obj_id,
        current=Shape("circle", {"cx": center[0], "cy": center[1], "r": radius}),
        fill=FlatFill(fill),
        z=0,
        footprint=poly,
    )


def test_clones_pass_proposes_use_for_translated_square_with_same_flat_fill():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#112233"))
    objects = [clone, canonical]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = optimize(objects, masks, [clones_pass])

    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"
    assert out[1].current.kind == "use"
    assert out[1].current.params == {
        "href_obj_id": 1,
        "transform": (1.0, 0.0, 0.0, 1.0, 34.0, 22.0),
        "fill": "#112233",
    }


def test_clones_pass_matches_rotated_congruent_square():
    canonical = _square(1, center=(22.0, 22.0), fill=FlatFill("#102030"))
    rotated = _square(2, center=(60.0, 48.0), angle_deg=30.0, fill=FlatFill("#102030"))
    objects = [canonical, rotated]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = optimize(objects, masks, [clones_pass])

    assert [obj.id for obj in out] == [1, 2]
    use_obj = out[1]
    assert use_obj.current.kind == "use"
    a, b, c, d, e, f = use_obj.current.params["transform"]
    assert a == pytest.approx(math.cos(math.radians(30.0)), abs=1e-6)
    assert b == pytest.approx(math.sin(math.radians(30.0)), abs=1e-6)
    assert c == pytest.approx(-math.sin(math.radians(30.0)), abs=1e-6)
    assert d == pytest.approx(math.cos(math.radians(30.0)), abs=1e-6)
    assert e == pytest.approx(51.94744111674235, abs=1e-6)
    assert f == pytest.approx(17.94744111674235, abs=1e-6)


def test_clones_pass_skips_non_congruent_shapes():
    square = _square(1, center=(20.0, 20.0))
    circle = _circle(2, center=(50.0, 20.0))
    objects = [square, circle]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = clones_pass(objects, masks)
    out = optimize(objects, masks, [clones_pass])

    assert proposals == []
    assert [obj.current.kind for obj in out] == ["path", "circle"]


def test_clones_pass_skips_recolored_clone_proposals():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#abcdef"))
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    assert clones_pass(objects, masks) == []
    out = optimize(objects, masks, [clones_pass])
    assert [obj.current.kind for obj in out] == ["path", "path"]


def test_clones_pass_skips_non_flat_fill_clone_proposals():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(
        2,
        center=(52.0, 40.0),
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    assert clones_pass(objects, masks) == []
    out = optimize(objects, masks, [clones_pass])
    assert [obj.current.kind for obj in out] == ["path", "path"]


def test_optimizer_object_svg_resolves_clone_href_to_emitted_id():
    canonical = _square(10, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(20, center=(52.0, 40.0), fill=FlatFill("#112233"))
    objects = [clone, canonical]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = optimize(objects, masks, [clones_pass])
    body = optimizer_objects_to_svg(out)

    assert body[0].startswith('<path id="s0"')
    assert body[1].startswith('<use id="s1"')
    assert 'href="#s0"' in body[1]
