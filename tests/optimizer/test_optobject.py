import numpy as np
from shapely.geometry import Polygon

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.occlusion import intersection_lens_d
from vectormark.optimizer.optobject import OptObject, VectorRegion, flatten_points, to_polygon


def test_to_polygon_circle_area_matches():
    circ = Shape("circle", {"cx": 50.0, "cy": 50.0, "r": 20.0})
    poly = to_polygon(circ, samples=64)
    assert abs(poly.area - np.pi * 20**2) / (np.pi * 20**2) < 0.01


def test_to_polygon_path_with_hole():
    d = "M0 0 L40 0 L40 40 L0 40 Z M10 10 L30 10 L30 30 L10 30 Z"
    poly = to_polygon(Shape("path", {"d": d}))
    assert abs(poly.area - (40 * 40 - 20 * 20)) < 4


def test_flatten_points_returns_outer_boundary_only_for_multi_subpath_shape():
    d = "M0 0 L40 0 L40 40 L0 40 Z M10 10 L30 10 L30 30 L10 30 Z"
    pts = flatten_points(Shape("path", {"d": d}))
    assert pts == [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]


def test_to_polygon_supports_absolute_arc_paths_from_intersection_lens():
    a = {"cx": 40.0, "cy": 50.0, "r": 24.0}
    b = {"cx": 60.0, "cy": 50.0, "r": 24.0}
    d = intersection_lens_d(a, b)
    assert d is not None

    poly = to_polygon(Shape("path", {"d": d}), samples=64)

    r = a["r"]
    dist = b["cx"] - a["cx"]
    theta = 2.0 * np.arccos(dist / (2.0 * r))
    expected = r * r * (theta - np.sin(theta))
    assert abs(poly.area - expected) / expected < 0.02


def test_optobject_with_exact_refreshes_flat():
    o = OptObject(
        id=1,
        exact=Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}),
        fill=FlatFill("#000000"),
        z=0,
    )
    assert abs(o.flat.area - 100) < 1
    o2 = o.with_exact(Shape("rect", {"x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0}))
    assert abs(o2.flat.area - 200) < 1 and abs(o.flat.area - 100) < 1


def test_optobject_is_vector_region_compatibility_constructor():
    raster = np.zeros((6, 6), dtype=bool)
    raster[1:5, 2:4] = True
    shape = Shape("rect", {"x": 2.0, "y": 1.0, "w": 2.0, "h": 4.0})

    region = OptObject(
        id=7,
        exact=shape,
        fill=FlatFill("#000000"),
        z=3,
        raster=raster,
        source_label=11,
        color_hex="#000000",
        diagnostics={"trace": {"area": int(raster.sum())}},
    )

    assert isinstance(region, VectorRegion)
    assert region.original == shape
    assert region.current == shape
    assert region.exact == shape
    assert np.array_equal(region.raster, raster)
    assert np.array_equal(region.mask, raster)
    assert abs(region.footprint.area - 8) < 1
    assert region.flat == region.footprint
    assert region.source_label == 11
    assert region.color_hex == "#000000"
    assert region.diagnostics["trace"]["area"] == 8
