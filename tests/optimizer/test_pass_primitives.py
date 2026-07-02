import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.trace import trace_regions
from vectormark.optimizer.framework import optimize
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.optimizer.passes.primitives import primitives_pass
from vectormark.pipeline import Options


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _disk_image():
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)
    return img


def _irregular_blob_image():
    img = np.full((120, 120, 3), 255, np.uint8)
    mask = np.zeros((120, 120), dtype=bool)
    mask[25:95, 25:95] = True
    mask[25:50, 25:50] = False
    mask[60:105, 70:95] = True
    img[mask] = (40, 40, 40)
    return img


def test_primitives_pass_proposes_circle_for_trace_disk_path():
    objects, masks = trace_regions(_disk_image(), Options())

    assert len(objects) == 1
    assert objects[0].current.kind == "path"

    proposals = primitives_pass(objects, masks)

    assert len(proposals) == 1
    assert proposals[0].obj_ids == (objects[0].id,)
    assert len(proposals[0].new_objects) == 1
    assert proposals[0].new_objects[0].current.kind == "circle"


def test_optimize_applies_primitives_pass_to_trace_disk():
    objects, masks = trace_regions(_disk_image(), Options())

    out = optimize(objects, masks, [primitives_pass])

    assert len(out) == 1
    assert out[0].current.kind == "circle"


def test_primitives_pass_skips_irregular_blob():
    objects, masks = trace_regions(_irregular_blob_image(), Options())

    assert len(objects) == 1
    assert objects[0].current.kind == "path"

    proposals = primitives_pass(objects, masks)
    out = optimize(objects, masks, [primitives_pass])

    assert proposals == []
    assert len(out) == 1
    assert out[0].current.kind == "path"


def test_primitives_pass_uses_largest_polygon_exterior_for_multipolygon():
    large = Point(40.0, 40.0).buffer(18.0, quad_segs=32)
    small = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)])
    obj = VectorRegion(
        id=7,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=MultiPolygon([small, large]),
    )

    proposals = primitives_pass([obj], {7: np.zeros((80, 80), dtype=bool)})

    assert len(proposals) == 1
    replacement = proposals[0].new_objects[0]
    assert replacement.current.kind == "circle"
    assert replacement.current.params["r"] > 10.0


def test_primitives_pass_skips_polygon_with_counter():
    outer = Point(40.0, 40.0).buffer(24.0, quad_segs=32)
    inner = Point(40.0, 40.0).buffer(5.0, quad_segs=32)
    ring = Polygon(outer.exterior.coords, [inner.exterior.coords])
    obj = VectorRegion(
        id=8,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=ring,
    )

    proposals = primitives_pass([obj], {8: np.zeros((80, 80), dtype=bool)})

    assert proposals == []


def test_primitives_pass_skips_empty_and_nonpolygon_geometry():
    empty = VectorRegion(
        id=1,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=GeometryCollection(),
    )
    nonpolygon = VectorRegion(
        id=2,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=Point(0.0, 0.0),
    )

    proposals = primitives_pass(
        [nonpolygon, empty],
        {
            1: np.zeros((4, 4), dtype=bool),
            2: np.zeros((4, 4), dtype=bool),
        },
    )

    assert proposals == []
