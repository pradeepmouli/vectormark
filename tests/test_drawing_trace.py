import numpy as np

from vectormark.drawing_trace import PythonTraceEngine, TraceOptions


def _annulus_image() -> np.ndarray:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:80, :80]
    ring = (12**2 <= (xx - 40) ** 2 + (yy - 40) ** 2) & ((xx - 40) ** 2 + (yy - 40) ** 2 <= 28**2)
    image[ring] = (0, 110, 240)
    return image


def _two_region_image() -> np.ndarray:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[12:38, 10:36] = (0, 110, 240)
    image[42:70, 44:72] = (255, 100, 0)
    return image


def test_trace_keeps_small_region_using_absolute_size_only():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[10:90, 10:90] = (0, 110, 240)
    image[4:8, 4:8] = (255, 100, 0)

    result = PythonTraceEngine().trace(image, TraceOptions(min_region_size=16))

    assert [region.id for region in result.regions] == ["r1", "r2"]


def test_trace_commands_are_scoped_by_region_and_subpath():
    region = PythonTraceEngine().trace(_annulus_image(), TraceOptions()).regions[0]

    assert region.trace_path.commands[0].id == "r1.p0.c0"
    assert any(command.id.startswith("r1.p1.") for command in region.trace_path.commands)


def test_region_map_labels_every_region():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())

    assert all(f">{region.id}<" in result.region_map_svg for region in result.regions)


def test_trace_result_public_dict_excludes_pixel_arrays():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())

    public = result.to_public_dict()

    assert public["width"] == 80
    assert public["regions"][0]["trace_path"]["commands"][0]["id"] == "r1.p0.c0"
    assert "mask" not in public["regions"][0]
