from __future__ import annotations

from .trace import _region_path_contours, _trace_shape, trace_regions

faithful_objects = trace_regions
_faithful_shape = _trace_shape

__all__ = [
    "_faithful_shape",
    "_region_path_contours",
    "faithful_objects",
    "trace_regions",
]
