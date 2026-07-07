from __future__ import annotations

from ..emit import apply_affine_point, shape_to_path_d, transform_path_d
from ..fit import Shape


def bake_shape_transform(shape: Shape, matrix: tuple[float, float, float, float, float, float]) -> Shape:
    if shape.kind == "circle":
        cx, cy = apply_affine_point(matrix, float(shape.params["cx"]), float(shape.params["cy"]))
        return Shape("circle", {"cx": cx, "cy": cy, "r": float(shape.params["r"])})
    if shape.kind == "path":
        params = dict(shape.params)
        params["d"] = transform_path_d(str(shape.params.get("d", "")), matrix)
        return Shape("path", params)
    return Shape("path", {"d": transform_path_d(shape_to_path_d(shape), matrix)})
