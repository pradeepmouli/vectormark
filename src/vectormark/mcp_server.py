"""MCP server for exposing vectormark logo idealization tools."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic import BaseModel, Field

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
from .drawing_refine import refine, root_scene
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
        "Idealize rendered raster logos into clean, editable SVG. "
        "Best inputs are flat-color marks, app icons, emblems, and simple logos."
    ),
)


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
    epsilon: float = Field(1.5, description="Primitive/polygon recognition tolerance in pixels.")
    max_error: float = Field(1.0, description="Bézier fit tolerance in pixels.")
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts, description="Server-side preprocessing options.")


class TraceDrawingOptions(BaseModel):
    refine: str = Field("interactive", description="interactive retains a live drawing; auto returns a one-shot idealization.")
    max_colors: int = 16
    min_region_size: int = 16
    trace_level: str = "pixel"
    simplify_tolerance: float = 1.5
    curve_tolerance: float = 1.0
    curve_type: str = "quadratic"
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts)


def _trace_result(image: ImageRef, options: TraceDrawingOptions):
    resolved = resolve_image(image.model_dump(exclude_none=True), local_trust=_LOCAL_TRUST)
    pre = options.preprocess
    rgb, _meta = preprocess_image(resolved.bytes, crop_to_content=pre.crop_to_content,
        max_size_px=pre.max_size_px, preserve_transparency=pre.preserve_transparency, quantize=pre.quantize)
    trace = PythonTraceEngine().trace(rgb, TraceOptions(max_colors=options.max_colors,
        min_region_size=options.min_region_size, trace_level=options.trace_level,
        simplify_tolerance=options.simplify_tolerance, curve_tolerance=options.curve_tolerance,
        curve_type=options.curve_type))
    return trace


@mcp.tool(title="Trace drawing", description="Trace a drawing into labeled raw paths. Interactive traces retain a live drawing for refine_drawing.")
def trace_drawing(image: ImageRef, options: TraceDrawingOptions | None = None, ctx: Context | None = None) -> CallToolResult:
    options = options or TraceDrawingOptions()
    if options.refine == "auto":
        return idealize_logo(image, IdealizeOptions(colors=options.max_colors, epsilon=options.simplify_tolerance,
            max_error=options.curve_tolerance, preprocess=options.preprocess))
    if options.refine != "interactive" or ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] interactive tracing requires an MCP session context")
    try:
        trace = _trace_result(image, options)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    drawing = _DRAWINGS.create(ctx.session, trace)
    scene = root_scene(trace)
    preview = _render_preview_png(scene.svg, trace.width, trace.height, [])
    result = {"drawing_id": drawing.id, "version": "v0", "trace": trace.to_public_dict(),
        "artifacts": {"svg": scene.svg, "labeled_svg": trace.region_map_svg}, "report": scene.report}
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if preview is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(title="Refine drawing", description="Apply a validated semantic plan to a live traced drawing and create a child version.")
def refine_drawing(plan: dict[str, object], ctx: Context | None = None) -> CallToolResult:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] refinement requires an MCP session context")
    try:
        parsed = parse_plan(plan)
        drawing, version = _DRAWINGS.get(ctx.session, parsed.drawing_id, parsed.base_version)
        base = version.scene if version.scene is not None else root_scene(drawing.trace)
        scene = refine(drawing.trace, base, parsed)
        child = _DRAWINGS.append(ctx.session, parsed.drawing_id, parsed.base_version, plan=plan, scene=scene, label=parsed.label)
    except (PlanValidationError, DrawingNotFound) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'PLAN_INVALID')}] {err}") from err
    preview = _render_preview_png(scene.svg, drawing.trace.width, drawing.trace.height, [])
    result = {"drawing_id": parsed.drawing_id, "version": child.id, "parent_version": parsed.base_version,
        "artifacts": {"svg": scene.svg}, "report": scene.report}
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if preview is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(
    title="Idealize logo",
    description=(
        "Idealize a raster logo into clean editable SVG. Pass the image by reference: a "
        "ChatGPT/host file (download_url+file_id), a local path, an https url, a data: URI, "
        "or base64. The server resolves and preprocesses it; no client-side base64 needed."
    ),
    meta={
        "openai/fileParams": ["image"],
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Idealizing logo...",
        "openai/toolInvocation/invoked": "Idealized logo.",
    },
)
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


@mcp.tool(
    title="Idealize logo from data",
    description=(
        "DEPRECATED fallback. Prefer `idealize_logo` with an image reference. Idealize a "
        "base64-encoded raster (bare base64 or a data:image/...;base64,... URI) into SVG, "
        "for hosts that cannot pass a file reference."
    ),
    meta={
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Idealizing image...",
        "openai/toolInvocation/invoked": "Idealized image.",
    },
)
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
    title="Render idealized logo",
    description="Render an idealize_logo result in the ChatGPT/MCP Apps widget. Pass the whole result object.",
    meta={
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Rendering SVG preview...",
        "openai/toolInvocation/invoked": "Rendered SVG preview.",
    },
)
def render_idealized_logo(result: dict | None = None, image_path: str = "", svg: str = "",
                          width: int = 0, height: int = 0) -> dict[str, object]:
    """Render an existing idealized SVG result in the vectormark app. Accepts the full
    `idealize_logo` result (preferred) or the legacy flat fields."""
    if result:
        svg = result.get("svg", svg)
        width = result.get("width", width)
        height = result.get("height", height)
        image_path = (result.get("diagnostics", {}).get("input", {}).get("source_kind")) or image_path
    return IdealizeLogoResult(
        image_path=image_path, output_path=None, width=width, height=height,
        svg_bytes=len(svg.encode()), svg=svg,   # svg_bytes re-derived, never trusted from caller
    ).to_dict()


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
