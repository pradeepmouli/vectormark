"""Stable, editable trace artifacts for the interactive drawing workflow."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

import numpy as np

from .color import extract_palette, quantize
from .contour import region_contours
from .emit import render_svg_doc
from .fit import fit_path
from .pipeline import attach_coverage_field
from .segment import segment


@dataclass(frozen=True)
class TraceOptions:
    max_colors: int = 16
    min_region_size: int = 16
    trace_level: Literal["pixel", "subpixel"] = "pixel"
    simplify_tolerance: float = 1.5
    curve_tolerance: float = 1.0
    curve_type: Literal["quadratic", "cubic"] = "quadratic"


@dataclass(frozen=True)
class TraceCommand:
    id: str
    command: Literal["M", "L", "Q", "C", "Z"]
    values: tuple[float, ...]


@dataclass(frozen=True)
class TracePath:
    d: str
    fill_rule: Literal["nonzero", "evenodd"]
    commands: tuple[TraceCommand, ...]


@dataclass(frozen=True)
class TraceRegion:
    id: str
    source_label: int
    color: str
    mask: np.ndarray = field(compare=False, repr=False)
    contours: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    trace_path: TracePath
    effective_trace_level: Literal["pixel", "subpixel"]


@dataclass(frozen=True)
class TraceResult:
    width: int
    height: int
    options: TraceOptions
    regions: tuple[TraceRegion, ...]
    region_map_svg: str

    def to_public_dict(self) -> dict[str, object]:
        """Return the stable, JSON-ready portion of the trace artifact."""
        return {
            "width": self.width,
            "height": self.height,
            "options": asdict(self.options),
            "regions": [
                {
                    "id": region.id,
                    "source_label": region.source_label,
                    "color": region.color,
                    "effective_trace_level": region.effective_trace_level,
                    "trace_path": {
                        "d": region.trace_path.d,
                        "fill_rule": region.trace_path.fill_rule,
                        "commands": [
                            {
                                "id": command.id,
                                "command": command.command,
                                "values": list(command.values),
                            }
                            for command in region.trace_path.commands
                        ],
                    },
                }
                for region in self.regions
            ],
            "region_map_svg": self.region_map_svg,
        }


class TraceEngine(Protocol):
    def trace(self, rgb: np.ndarray, options: TraceOptions) -> TraceResult: ...


_PATH_TOKEN = re.compile(r"[MLQCZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_COORDINATE_COUNTS = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}


def _path_commands(d: str, region_id: str, subpath_index: int) -> tuple[TraceCommand, ...]:
    tokens = _PATH_TOKEN.findall(d)
    commands: list[TraceCommand] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if command not in _COORDINATE_COUNTS:
            raise ValueError(f"unsupported path data in trace: {d!r}")
        index += 1
        count = _COORDINATE_COUNTS[command]
        values = tuple(float(token) for token in tokens[index:index + count])
        if len(values) != count:
            raise ValueError(f"incomplete {command} command in trace: {d!r}")
        index += count
        commands.append(
            TraceCommand(
                id=f"{region_id}.p{subpath_index}.c{len(commands)}",
                command=command,  # type: ignore[arg-type]
                values=values,
            )
        )
    return tuple(commands)


def _bbox_origin(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min())


def _region_map(width: int, height: int, regions: tuple[TraceRegion, ...]) -> str:
    body: list[str] = []
    for region in regions:
        ys, xs = np.nonzero(region.mask)
        cx = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
        cy = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
        body.append(
            f'<path id="map-{region.id}" fill="{region.color}" fill-opacity="0.35" '
            f'fill-rule="{region.trace_path.fill_rule}" d="{region.trace_path.d}"/>'
        )
        body.append(
            f'<text x="{cx:g}" y="{cy:g}" text-anchor="middle" '
            f'dominant-baseline="middle">{region.id}</text>'
        )
    return render_svg_doc(width, height, body)


class PythonTraceEngine:
    """Pure-Python adapter from an RGB image to a deterministic trace artifact."""

    def trace(self, rgb: np.ndarray, options: TraceOptions) -> TraceResult:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("trace expects an RGB image with shape (height, width, 3)")
        height, width, _ = image.shape
        # Interactive tracing is governed by the explicit absolute region floor;
        # a relative palette floor must not erase a region before segmentation.
        palette = extract_palette(image, max_colors=options.max_colors, min_fraction=0.0)
        regions = segment(quantize(image, palette), min_area=options.min_region_size)
        if options.trace_level == "subpixel":
            attach_coverage_field(regions, image, options.max_colors, palette=palette)

        ordered = sorted(
            regions,
            key=lambda region: (-region.area, *_bbox_origin(region.mask), region.color_hex, region.label),
        )
        traced: list[TraceRegion] = []
        for position, region in enumerate(ordered, start=1):
            region_id = f"r{position}"
            effective_level: Literal["pixel", "subpixel"] = (
                "subpixel" if options.trace_level == "subpixel" and region.coverage is not None else "pixel"
            )
            contours = tuple(
                contour
                for contour in region_contours(
                    region.mask,
                    coverage=region.coverage if effective_level == "subpixel" else None,
                )
                if len(contour) >= 3
            )
            path_ds = [
                fit_path(
                    contour,
                    epsilon=options.simplify_tolerance,
                    max_error=options.curve_tolerance,
                    cubic=options.curve_type == "cubic",
                ).params["d"]
                for contour in contours
            ]
            commands = tuple(
                command
                for subpath_index, d in enumerate(path_ds)
                for command in _path_commands(d, region_id, subpath_index)
            )
            mask = region.mask.copy()
            mask.setflags(write=False)
            immutable_contours = tuple(contour.copy() for contour in contours)
            for contour in immutable_contours:
                contour.setflags(write=False)
            traced.append(
                TraceRegion(
                    id=region_id,
                    source_label=region.label,
                    color=region.color_hex,
                    mask=mask,
                    contours=immutable_contours,
                    trace_path=TracePath(
                        d=" ".join(path_ds),
                        fill_rule="evenodd" if len(path_ds) > 1 else "nonzero",
                        commands=commands,
                    ),
                    effective_trace_level=effective_level,
                )
            )
        result_regions = tuple(traced)
        return TraceResult(
            width=width,
            height=height,
            options=options,
            regions=result_regions,
            region_map_svg=_region_map(width, height, result_regions),
        )
