"""MCP server for exposing vectormark logo idealization tools."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .mcp_image import (
    DEFAULT_COLORS,
    DEFAULT_MAX_SIZE_PX,
    ImageError,
    preprocess_image,
    resolve_image,
    svg_output_facts,
)
from .pipeline import Options, _flatten_on_white, idealize
from .drawing_plan import PlanValidationError, parse_plan
from .drawing_refine import (
    drawing_summary,
    labeled_drawing_svg,
    refine,
    render_drawing as render_drawing_regions,
    root_regions,
    stitch_regions,
)
from .drawing_state import DrawingNotFound, DrawingStore
from .drawing_trace import PythonTraceEngine, TraceOptions

# Transport is chosen at startup. stdio = local, full-trust (your own machine, your
# files). Any HTTP transport is potentially network-reachable, so the filesystem tools
# (arbitrary-path READ via idealize_logo, WRITE via output_path) are withheld and only
# the byte-in/SVG-out tools are exposed. See docs/mcp.md.
_TRANSPORT = (os.environ.get("VECTORMARK_MCP_TRANSPORT") or "stdio").strip()
_LOCAL_TRUST = _TRANSPORT == "stdio"

WIDGET_URI = "ui://vectormark/logo-widget.html"
_DRAWINGS = DrawingStore()
_WIDGET_HTML_PATH = Path(__file__).parents[2] / "integrations" / "mcp-app" / "dist" / "mcp-app.html"
_WIDGET_BUILD_REQUIRED_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>vectormark</title>
</head>
<body>
  <main id="root">
    <h1>Logo idealizer</h1>
    <p>vectormark MCP app has not been built yet. Run <code>npm --prefix integrations/mcp-app run build</code>.</p>
  </main>
</body>
</html>
""".strip()


def _read_widget_html() -> str:
    if _WIDGET_HTML_PATH.is_file():
        return _WIDGET_HTML_PATH.read_text()
    return _WIDGET_BUILD_REQUIRED_HTML


@dataclass(frozen=True)
class IdealizeLogoResult:
    """Structured result returned by the MCP tool and testable helper."""

    image_path: str
    output_path: str | None
    width: int
    height: int
    svg_bytes: int
    svg: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def idealize_logo_file(
    image_path: str,
    *,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> IdealizeLogoResult:
    """Idealize one raster logo file and optionally write the SVG to disk."""

    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input raster does not exist: {source}")

    options = Options(
        epsilon=epsilon,
        max_error=max_error,
        max_colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    )
    svg = idealize(str(source), options=options)

    destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg)

    with Image.open(source) as image:
        width, height = image.size

    return IdealizeLogoResult(
        image_path=str(source),
        output_path=str(destination) if destination is not None else None,
        width=width,
        height=height,
        svg_bytes=len(svg.encode()),
        svg=svg,
    )


def _decode_image_base64(image_base64: str) -> bytes:
    """Decode a base64 raster, tolerating a `data:image/...;base64,` URI prefix.
    Raises ValueError on invalid base64."""
    data = image_base64.strip()
    if data.startswith("data:"):
        _, _, data = data.partition(",")   # drop the data-URI media-type prefix
    try:
        return base64.b64decode(data, validate=True)   # binascii.Error is a ValueError
    except ValueError as exc:
        raise ValueError("image_base64 is not valid base64") from exc


def idealize_logo_bytes(
    image_bytes: bytes,
    *,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> IdealizeLogoResult:
    """Idealize an in-memory raster (e.g. an image an agent just generated) into SVG,
    with no local file needed. Alpha is composited on white via the shared pipeline
    helper, matching file input."""

    arr = _flatten_on_white(Image.open(io.BytesIO(image_bytes)))
    height, width = arr.shape[:2]
    options = Options(
        epsilon=epsilon,
        max_error=max_error,
        max_colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    )
    svg = idealize(arr, options=options)

    destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg)

    return IdealizeLogoResult(
        image_path="(inline base64 image)",
        output_path=str(destination) if destination is not None else None,
        width=width,
        height=height,
        svg_bytes=len(svg.encode()),
        svg=svg,
    )


def _render_preview_png(svg: str, width: int, height: int, warnings: list[str]) -> bytes | None:
    """Best-effort: render the SVG to PNG via resvg-py; None (+ a warning) if unavailable."""
    try:
        import resvg_py
        return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))
    except Exception as exc:
        warnings.append(f"preview unavailable: {exc}")
        return None


def idealize_logo_image(
    image: dict, options: dict | None, *, local_trust: bool
) -> tuple[dict, bytes | None]:
    """Resolve -> preprocess -> idealize a referenced image. Returns (structured_result,
    preview_png_bytes|None). Raises ImageError with a structured code on input failure."""
    options = options or {}
    pre = options.get("preprocess") or {}
    resolved = resolve_image(image, local_trust=local_trust)

    arr, meta = preprocess_image(
        resolved.bytes,
        crop_to_content=pre.get("crop_to_content", True),
        max_size_px=pre.get("max_size_px", DEFAULT_MAX_SIZE_PX),
        preserve_transparency=pre.get("preserve_transparency", True),
        quantize=pre.get("quantize", False),
    )

    opts = Options(
        epsilon=options.get("epsilon", 1.5),
        max_error=options.get("max_error", 1.0),
        max_colors=options.get("colors", DEFAULT_COLORS),
        flatten=options.get("flatten", False),
        no_symmetry=options.get("no_symmetry", False),
        optimizer=options.get("optimizer", False),
    )
    try:
        svg = idealize(arr, options=opts)
    except Exception as exc:
        raise ImageError(
            "VECTORMARK_FAILED", f"vectormark failed to process the image: {exc}"
        ) from exc

    with Image.open(io.BytesIO(resolved.bytes)) as src:
        ow, oh = src.size
    warnings: list[str] = []
    preview = _render_preview_png(svg, meta.width, meta.height, warnings)
    _svg_bytes = len(svg.encode())

    diagnostics = {
        "input": {
            "source_kind": resolved.source_kind,
            "mime_type": resolved.mime_type,
            "bytes": len(resolved.bytes),
            "sha256": resolved.sha256,
            "original_width": ow,
            "original_height": oh,
        },
        "processed": {
            "width": meta.width,
            "height": meta.height,
            "cropped": meta.cropped,
            "resized": meta.resized,
            "transparent": meta.transparent,
            "quantized": meta.quantized,
        },
        "vectormark": {
            "colors": opts.max_colors,
            "flatten": opts.flatten,
            "no_symmetry": opts.no_symmetry,
            "epsilon": opts.epsilon,
            "max_error": opts.max_error,
        },
        "output": {"svg_bytes": _svg_bytes, **svg_output_facts(svg)},
        "warnings": warnings,
    }
    result = {
        "svg": svg,
        "width": meta.width,
        "height": meta.height,
        "svg_bytes": _svg_bytes,
        "preview_available": preview is not None,
        "diagnostics": diagnostics,
    }
    return result, preview


mcp = FastMCP(
    "vectormark",
    instructions=(
        "Convert rendered raster artwork into clean, editable SVG. Use trace_drawing with "
        "refine='auto' for one-shot idealization. Use refine='none' to retain unrefined trace roots, "
        "inspect the returned preview, labeled_svg, and raw_trace artifacts, then make semantic "
        "judgments about every retained region (primitive, path geometry, symmetry, clones, fill, "
        "z-order, simplify, and stitch). Submit refine_drawing plans with the drawing_id and an "
        "existing base_version; every plan creates a branch version. Preserve a path when no supported "
        "operation improves it. Path-local operations use command IDs exposed on each retained target geometry."
    ),
)


class _PlanSchema(BaseModel):
    """Static refinement-plan contract exposed to MCP clients.

    Live drawing validation remains in ``drawing_plan``: target IDs, trace command
    ownership, command contiguity, and version rules need the traced drawing.
    """

    model_config = ConfigDict(extra="forbid")


class PlanDefaults(_PlanSchema):
    epsilon: float | None = Field(None, description="Primitive/polygon fitting tolerance in pixels.")
    max_error: float | None = Field(None, description="Maximum path-fit residual in pixels.")


class AxisSpec(_PlanSchema):
    theta: float = Field(description="Symmetry-axis angle in radians.")
    cx: float = Field(description="X coordinate of a point on the symmetry axis.")
    cy: float = Field(description="Y coordinate of a point on the symmetry axis.")


class FlatFillSpec(_PlanSchema):
    type: Literal["flat"]
    color: str = Field(description="CSS #RRGGBB fill color.")


class LinearGradientFillSpec(_PlanSchema):
    type: Literal["linear_gradient"]
    geometry: dict[str, float] = Field(description="Gradient endpoints: x1, y1, x2, y2.")
    stops: list[tuple[float, str]] = Field(description="Ordered [offset, #RRGGBB] gradient stops.")


class RadialGradientFillSpec(_PlanSchema):
    type: Literal["radial_gradient"]
    geometry: dict[str, float] = Field(description="Radial gradient geometry: cx, cy, r.")
    stops: list[tuple[float, str]] = Field(description="Ordered [offset, #RRGGBB] gradient stops.")


class RasterFillSpec(_PlanSchema):
    type: Literal["raster"]
    geometry: dict[str, float] = Field(description="Raster placement: x, y, w, h.")
    png_b64: str = Field(description="Base64-encoded PNG data.")


FillSpec = Annotated[
    FlatFillSpec | LinearGradientFillSpec | RadialGradientFillSpec | RasterFillSpec,
    Field(discriminator="type"),
]


class PrimitiveGeometrySpec(_PlanSchema):
    type: Literal["circle", "ellipse", "rect", "rounded_rect", "polygon", "trapezoid", "rounded_trapezoid", "cap"]


class PathFitOp(_PlanSchema):
    op: Literal["fit"]
    target: str = Field(description="A retained, version-scoped segment ID, for example r1.p1.c14-1.")
    type: Literal["line", "quadratic", "cubic", "keep"] = Field(
        description="Geometry for this segment. Use line for deliberate straight edges; keep preserves its retained command."
    )


class PathMatchLengthOp(_PlanSchema):
    op: Literal["match_length"]
    target: str = Field(description="Retained, version-scoped segment to resize.")
    reference: str = Field(description="Retained, version-scoped segment whose length is matched.")


class PathMatchOp(_PlanSchema):
    op: Literal["match"]
    target: str = Field(description="Retained, version-scoped segment to reconstruct.")
    reference: str = Field(description="Retained, version-scoped segment to copy.")
    transform: tuple[float, float, float, float, float, float] = Field(
        description="SVG affine transform [a, b, c, d, e, f] from the reference segment to the target segment."
    )


class PathSetParallelOp(_PlanSchema):
    op: Literal["set_parallel"]
    target: str = Field(description="Retained, version-scoped segment to fit as a parallel line.")
    reference: str = Field(description="Retained segment supplying the supporting-line direction, never its length.")
    distance: float | None = Field(None, description="Optional signed perpendicular offset. Omit to fit the offset from the target's existing geometry.")


class PathAlignOp(_PlanSchema):
    op: Literal["align"]
    target: str = Field(description="Retained, version-scoped segment whose endpoint is aligned.")
    reference: str = Field(description="Retained, version-scoped segment supplying the endpoint coordinate.")
    axes: list[Literal["x", "y"]] = Field(min_length=1, description="Endpoint axes to align: x, y, or both.")


class PathRemoveOp(_PlanSchema):
    op: Literal["remove"]
    target: str = Field(description="Retained, version-scoped segment to remove while preserving following geometry.")


class PathBreakOp(_PlanSchema):
    op: Literal["break"]
    target: str = Field(description="The current fitted segment to end with a sharp discontinuity.")


class PathCloseOp(_PlanSchema):
    op: Literal["close"]


class PathSimplifyOp(_PlanSchema):
    op: Literal["simplify"]


class PathStitchOp(_PlanSchema):
    op: Literal["stitch"]


PathOp = Annotated[
    PathFitOp | PathMatchLengthOp | PathMatchOp | PathSetParallelOp | PathAlignOp | PathRemoveOp | PathBreakOp | PathCloseOp | PathSimplifyOp | PathStitchOp,
    Field(discriminator="op"),
]


class PathGeometrySpec(_PlanSchema):
    type: Literal["path"]
    ops: list[PathOp] = Field(
        min_length=1,
        description="Path-local program. Fit retained segments, then optionally break, close, simplify, or stitch.",
    )


GeometrySpec = Annotated[PrimitiveGeometrySpec | PathGeometrySpec, Field(discriminator="type")]


class _ToleranceOp(_PlanSchema):
    epsilon: float | None = Field(None, description="Per-operation fitting tolerance in pixels.")
    max_error: float | None = Field(None, description="Per-operation maximum fit residual in pixels.")


class MergeOp(_PlanSchema):
    op: Literal["merge"]
    id: str = Field(description="New semantic group ID.")
    regions: list[str] = Field(min_length=1, description="Current retained region IDs to union into the new group.")


class SplitOp(_PlanSchema):
    op: Literal["split"]
    target: str = Field(description="Current target to split; must be the final operation in a plan.")


class DetectPrimitivesOp(_ToleranceOp):
    op: Literal["detect_primitives"]
    target: str | None = Field(None, description="Optional target; omit to inspect all current targets.")


class DetectSymmetryOp(_ToleranceOp):
    op: Literal["detect_symmetry"]
    target: str | None = Field(None, description="Optional target; symmetry operations must be terminal.")


class DetectClonesOp(_ToleranceOp):
    op: Literal["detect_clones"]
    target: str | None = Field(None, description="Optional target; omit to inspect all current targets.")


class SimplifyOp(_ToleranceOp):
    op: Literal["simplify"]
    target: str | None = Field(None, description="Optional current target; omit to simplify all current targets.")


class StitchOp(_ToleranceOp):
    op: Literal["stitch"]
    target: str | None = Field(None, description="Optional current target; omit to reconcile all shared boundaries.")


class SetGeometryOp(_ToleranceOp):
    op: Literal["set_geometry"]
    target: str = Field(description="Current retained region or merged group ID.")
    geometry: GeometrySpec


class SetFillOp(_PlanSchema):
    op: Literal["set_fill"]
    target: str = Field(description="Current retained region or merged group ID.")
    fill: FillSpec


class SetZOrderOp(_PlanSchema):
    op: Literal["set_z_order"]
    targets: list[str] = Field(min_length=1, description="All current targets in back-to-front paint order.")


class CloneOp(_PlanSchema):
    op: Literal["clone"]
    source: str = Field(description="Existing source target ID.")
    target: str = Field(description="Existing target ID replaced by a transformed clone.")
    transform: tuple[float, float, float, float, float, float] = Field(
        description="SVG affine transform [a, b, c, d, e, f] from source to target."
    )


class AlignOp(_PlanSchema):
    op: Literal["align"]
    target: str = Field(description="Current retained region to translate.")
    reference: str = Field(description="Current retained region whose bounds center is used as the reference.")
    axes: list[Literal["x", "y"]] = Field(min_length=1, description="Axes to align: x, y, or both.")


class SetSymmetryOp(_PlanSchema):
    op: Literal["set_symmetry"]
    source: str = Field(description="Existing source target ID.")
    target: str = Field(description="Existing target ID replaced by source mirrored around axis.")
    axis: AxisSpec


PlanOp = Annotated[
    MergeOp | SplitOp | DetectPrimitivesOp | DetectSymmetryOp | DetectClonesOp | SimplifyOp | StitchOp
    | SetGeometryOp | SetFillOp | SetZOrderOp | CloneOp | AlignOp | SetSymmetryOp,
    Field(discriminator="op"),
]


class RefinementPlanInput(_PlanSchema):
    """The statically typed public contract for ``refine_drawing``."""

    version: Literal["vectormark.plan.v1"]
    drawing_id: str = Field(min_length=1, description="Live drawing ID returned by trace_drawing.")
    base_version: str = Field(pattern=r"^v\d+(?:\.\d+)*$", description="Existing version to branch from, such as v0 or v0.1.")
    label: str | None = Field(None, max_length=200, description="Optional human-readable branch label.")
    defaults: PlanDefaults = Field(default_factory=PlanDefaults, description="Default fitting tolerances for operations.")
    ops: list[PlanOp] = Field(
        description="Ordered semantic transformations. IDs refer to the selected base version; path geometry uses retained segment IDs such as r1.p1.c14-1.",
        json_schema_extra={
            "examples": [[
                {"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}},
                {"op": "detect_symmetry", "target": "r1"},
            ]]
        },
    )


class _DrawingOutputSchema(BaseModel):
    """Stable structured result fields returned by drawing-first tools."""

    model_config = ConfigDict(extra="forbid")


class DrawingArtifactsOutput(_DrawingOutputSchema):
    svg: str
    preview: str
    labeled_svg: str
    raw_trace: str


class DrawingTargetOutput(_DrawingOutputSchema):
    id: str
    source_regions: list[str]
    geometry: str
    fill: str
    z: float
    diagnostics: dict[str, Any]


class DrawingReportOutput(_DrawingOutputSchema):
    targets: list[DrawingTargetOutput]


class TraceRegionOutput(_DrawingOutputSchema):
    id: str
    source_regions: list[str]
    geometry: dict[str, Any]
    fill: dict[str, Any]


class TraceSummaryOutput(_DrawingOutputSchema):
    width: int
    height: int
    options: dict[str, Any]
    regions: list[TraceRegionOutput]


class TraceDrawingOutput(_DrawingOutputSchema):
    """One output envelope for auto-refined and unrefined trace modes."""

    drawing_id: str | None = None
    version: str | None = None
    trace: TraceSummaryOutput | None = None
    artifacts: DrawingArtifactsOutput | None = None
    report: DrawingReportOutput | None = None
    svg: str | None = None
    width: int | None = None
    height: int | None = None
    svg_bytes: int | None = None
    preview_available: bool | None = None
    diagnostics: dict[str, Any] | None = None


class RefinedDrawingOutput(_DrawingOutputSchema):
    drawing_id: str
    version: str
    parent_version: str
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class DrawingArtifactOutput(_DrawingOutputSchema):
    """Artifact envelope; fields vary predictably by the typed MIME value."""

    mime_type: Literal["image/svg+xml", "application/json", "image/png"]
    svg: str | None = None
    trace: dict[str, Any] | None = None


class RenderedSvgOutput(_DrawingOutputSchema):
    image_path: str
    output_path: str | None
    width: int
    height: int
    svg_bytes: int
    svg: str


class ImageRef(BaseModel):
    """A reference to the source image. Provide exactly one source. ChatGPT/host file
    params fill `download_url` (+ `file_id`); callers may instead pass a `path`, https
    `url`, `data_uri`, or bare `base64`."""

    download_url: str | None = Field(
        None, description="Temporary URL to GET the file bytes (ChatGPT/host file param)."
    )
    file_id: str | None = Field(None, description="Host file identifier accompanying download_url.")
    mime_type: str | None = Field(None, description="Optional declared MIME type of the file.")
    file_name: str | None = Field(None, description="Optional original file name.")
    path: str | None = Field(
        None, description="Local filesystem path. Resolved only on the stdio/local-trust server; rejected over HTTP."
    )
    url: str | None = Field(None, description="Explicit https URL to fetch (SSRF-guarded).")
    data_uri: str | None = Field(None, description="A data:image/...;base64,... URI.")
    base64: str | None = Field(None, description="Bare base64-encoded image bytes (PNG/JPEG/WebP).")


class PreprocessOpts(BaseModel):
    """Server-side preprocessing applied before idealization."""

    crop_to_content: bool = Field(True, description="Trim transparent/near-white margins to the content bounding box.")
    max_size_px: int = Field(DEFAULT_MAX_SIZE_PX, description="Downscale (never upscale) so the longer side is at most this many px.")
    preserve_transparency: bool = Field(True, description="Keep alpha through cropping; composite on white at the end.")
    quantize: bool = Field(False, description="Pre-quantize the raster (usually harmful; vectormark extracts its own palette).")


class IdealizeOptions(BaseModel):
    """Idealization parameters. All optional with sensible defaults."""

    colors: int = Field(DEFAULT_COLORS, description="Max palette colors. A CEILING, not a target — flats stay flat; raise it to let gradients keep their bands.")
    flatten: bool = Field(False, description="Emit plain paths instead of native SVG primitives and <use> mirror.")
    no_symmetry: bool = Field(False, description="Disable symmetry detection.")
    optimizer: bool = Field(False, description="Run the existing geometry optimizer passes.")
    epsilon: float = Field(1.5, description="Primitive/polygon recognition tolerance in pixels.")
    max_error: float = Field(1.0, description="Bézier fit tolerance in pixels.")
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts, description="Server-side preprocessing options.")


class TraceDrawingOptions(BaseModel):
    refine: Literal["auto", "none"] = Field(
        "none", description="Automatic refinement at trace time: none retains trace roots; auto returns a one-shot idealization."
    )
    max_colors: int = Field(16, ge=2, description="Maximum quantized palette colors used by the raw trace.")
    min_region_size: int = Field(16, ge=1, description="Absolute pixel-area floor for raw trace regions.")
    min_region_fraction: float = Field(0.02, ge=0, lt=1, description="Relative area floor for retained trace roots; raw trace remains available on demand.")
    trace_level: Literal["pixel", "subpixel"] = Field(
        "pixel", description="Boundary trace precision. subpixel uses anti-alias coverage when available."
    )
    simplify_tolerance: float = Field(1.5, ge=0, description="Raw contour simplification tolerance in pixels.")
    curve_tolerance: float = Field(1.0, ge=0, description="Raw quadratic/cubic path-fit tolerance in pixels.")
    curve_type: Literal["quadratic", "cubic"] = Field("quadratic", description="Raw trace curve representation.")
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts)


def _trace_result(image: ImageRef, options: TraceDrawingOptions):
    resolved = resolve_image(image.model_dump(exclude_none=True), local_trust=_LOCAL_TRUST)
    pre = options.preprocess
    # Raw region tracing needs a border/background plate; cropping it away makes a
    # single-colour drawing indistinguishable from the background.
    rgb, _meta = preprocess_image(resolved.bytes, crop_to_content=False,
        max_size_px=pre.max_size_px, preserve_transparency=pre.preserve_transparency, quantize=pre.quantize)
    trace = PythonTraceEngine().trace(rgb, TraceOptions(max_colors=options.max_colors,
        min_region_size=options.min_region_size, trace_level=options.trace_level,
        simplify_tolerance=options.simplify_tolerance, curve_tolerance=options.curve_tolerance,
        curve_type=options.curve_type))
    return trace, rgb


@mcp.tool(
    title="Trace drawing",
    description="Trace a drawing into labeled raw paths. Interactive traces retain a live drawing for refine_drawing.",
    meta={"openai/fileParams": ["image"]},
)
def trace_drawing(
    image: ImageRef, options: TraceDrawingOptions | None = None, ctx: Context | None = None
) -> Annotated[CallToolResult, TraceDrawingOutput]:
    options = options or TraceDrawingOptions()
    if options.refine == "auto":
        return idealize_logo(image, IdealizeOptions(colors=options.max_colors, epsilon=options.simplify_tolerance,
            max_error=options.curve_tolerance, optimizer=True, preprocess=options.preprocess))
    if options.refine != "none" or ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] refine='none' tracing requires an MCP session context")
    try:
        trace, rgb = _trace_result(image, options)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    regions = stitch_regions(trace, root_regions(trace, rgb, min_region_fraction=options.min_region_fraction))
    drawing = _DRAWINGS.create(ctx.session, trace, regions=regions)
    rendered = render_drawing_regions(trace, regions)
    preview = _render_preview_png(rendered.svg, trace.width, trace.height, [])
    artifact_base = f"drawing://{drawing.id}/v0"
    result = {"drawing_id": drawing.id, "version": "v0", "trace": drawing_summary(trace, regions),
        "artifacts": {"svg": artifact_base + ".svg", "preview": artifact_base + ".png", "labeled_svg": artifact_base + ".labels.svg", "raw_trace": artifact_base + ".trace.json"}, "report": dict(rendered.report)}
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if preview is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(
    title="Refine drawing",
    description=(
        "Apply an ordered, strongly typed semantic plan to a live traced drawing and create a child version. "
        "Inspect trace_drawing artifacts first: labeled_svg identifies retained targets and each target geometry supplies command IDs for "
        "path geometry. Choose only improvements that preserve the design intent; omitting an operation preserves the target."
    ),
)
def refine_drawing(
    plan: RefinementPlanInput, ctx: Context | None = None
) -> Annotated[CallToolResult, RefinedDrawingOutput]:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] refinement requires an MCP session context")
    try:
        plan_payload = plan.model_dump(exclude_none=True) if isinstance(plan, RefinementPlanInput) else plan
        parsed = parse_plan(plan_payload)
        drawing, version = _DRAWINGS.get(ctx.session, parsed.drawing_id, parsed.base_version)
        assert version.regions is not None
        base = version.regions
        regions = refine(drawing.trace, base, parsed)
        child = _DRAWINGS.append(
            ctx.session, parsed.drawing_id, parsed.base_version, plan=plan_payload, regions=regions, label=parsed.label
        )
    except (PlanValidationError, DrawingNotFound, ValueError, KeyError) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'PLAN_INVALID')}] {err}") from err
    rendered = render_drawing_regions(drawing.trace, regions)
    preview = _render_preview_png(rendered.svg, drawing.trace.width, drawing.trace.height, [])
    artifact_base = f"drawing://{parsed.drawing_id}/{child.id}"
    result = {"drawing_id": parsed.drawing_id, "version": child.id, "parent_version": parsed.base_version,
        "artifacts": {"svg": artifact_base + ".svg", "preview": artifact_base + ".png", "labeled_svg": artifact_base + ".labels.svg", "raw_trace": artifact_base + ".trace.json"}, "report": dict(rendered.report)}
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if preview is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(title="Get drawing artifact", description="Fetch an SVG, clean PNG preview, or labeled trace SVG for a live drawing version.")
def get_drawing_artifact(
    drawing_id: str, version: str = "v0", artifact: Literal["svg", "preview", "preview_png", "labeled_svg", "raw_trace"] = "svg",
    ctx: Context | None = None,
) -> Annotated[CallToolResult, DrawingArtifactOutput]:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] artifact retrieval requires an MCP session context")
    try:
        drawing, stored = _DRAWINGS.get(ctx.session, drawing_id, version)
    except DrawingNotFound as err:
        raise ToolError("[DRAWING_NOT_FOUND] drawing or version is unavailable") from err
    assert stored.regions is not None
    regions = stored.regions
    rendered = render_drawing_regions(drawing.trace, regions)
    if artifact == "labeled_svg":
        payload = labeled_drawing_svg(drawing.trace, regions)
        return CallToolResult(content=[TextContent(type="text", text=payload)], structuredContent={"mime_type": "image/svg+xml", "svg": payload})
    if artifact == "raw_trace":
        payload = drawing.trace.to_public_dict(include_region_map=True)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent={"mime_type": "application/json", "trace": payload},
        )
    if artifact == "svg":
        return CallToolResult(content=[TextContent(type="text", text=rendered.svg)], structuredContent={"mime_type": "image/svg+xml", "svg": rendered.svg})
    if artifact in {"preview", "preview_png"}:
        preview = _render_preview_png(rendered.svg, drawing.trace.width, drawing.trace.height, [])
        if preview is None:
            raise ToolError("[PREVIEW_UNAVAILABLE] PNG preview renderer is unavailable")
        content = ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png")
        return CallToolResult(content=[content], structuredContent={"mime_type": "image/png"})
    raise ToolError("[ARTIFACT_UNKNOWN] artifact must be svg, preview, preview_png, labeled_svg, or raw_trace")


def idealize_logo(image: ImageRef, options: IdealizeOptions | None = None) -> CallToolResult:
    """File-first logo idealization. Returns structured content plus a best-effort image block."""
    image_dict = image.model_dump(exclude_none=True)
    options_dict = options.model_dump() if options is not None else None
    try:
        result, preview = idealize_logo_image(image_dict, options_dict, local_trust=_LOCAL_TRUST)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(result))
    ]
    if preview is not None:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(preview).decode(),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result, isError=False)


def idealize_logo_data(
    image_base64: str,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> dict[str, object]:
    """Convert a base64-encoded raster image into structured SVG."""

    # No host-filesystem writes unless local-trust (stdio): drop output_path otherwise.
    return idealize_logo_bytes(
        _decode_image_base64(image_base64),
        output_path=output_path if _LOCAL_TRUST else None,
        epsilon=epsilon,
        max_error=max_error,
        colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    ).to_dict()


@mcp.tool(
    title="Render drawing",
    description="Render a drawing SVG in the ChatGPT/MCP Apps widget.",
    meta={
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Rendering SVG preview...",
        "openai/toolInvocation/invoked": "Rendered SVG preview.",
    },
)
def render_drawing(result: dict | None = None, image_path: str = "", svg: str = "",
                          width: int = 0, height: int = 0) -> RenderedSvgOutput:
    """Render an existing traced/refined drawing SVG in the vectormark app.

    Accepts a drawing-tool result (preferred) or the legacy flat SVG fields.
    """
    if result:
        svg = result.get("svg", svg)
        width = result.get("width", width)
        height = result.get("height", height)
        image_path = (result.get("diagnostics", {}).get("input", {}).get("source_kind")) or image_path
    return RenderedSvgOutput.model_validate(IdealizeLogoResult(
        image_path=image_path, output_path=None, width=width, height=height,
        svg_bytes=len(svg.encode()), svg=svg,   # svg_bytes re-derived, never trusted from caller
    ).to_dict()).model_dump()


@mcp.resource(
    WIDGET_URI,
    name="vectormark-logo-widget",
    title="vectormark logo preview",
    description="Preview and rerun vectormark logo idealization.",
    mime_type="text/html;profile=mcp-app",
    meta={
        "ui": {
            "prefersBorder": True,
            "csp": {"connectDomains": [], "resourceDomains": []},
        },
        "openai/widgetDescription": "Interactive preview for a vectormark SVG result.",
        "openai/widgetPrefersBorder": True,
        "openai/widgetCSP": {
            "connect_domains": [],
            "resource_domains": [],
        },
    },
)
def logo_widget() -> str:
    """Return the HTML component used by MCP Apps hosts."""

    return _read_widget_html()


def main() -> None:
    """Run the server. Transport from VECTORMARK_MCP_TRANSPORT (stdio | sse |
    streamable-http; default stdio). For HTTP transports, VECTORMARK_MCP_HOST /
    VECTORMARK_MCP_PORT set the bind address (default 127.0.0.1:8000; the MCP endpoint
    is served at /mcp)."""
    if _TRANSPORT != "stdio":
        host = os.environ.get("VECTORMARK_MCP_HOST")
        port = os.environ.get("VECTORMARK_MCP_PORT")
        if host:
            mcp.settings.host = host
        if port:
            mcp.settings.port = int(port)
    mcp.run(_TRANSPORT)


if __name__ == "__main__":
    main()
