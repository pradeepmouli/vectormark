"""MCP server for exposing vectormark logo idealization tools."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

from .mcp_image import (
    DEFAULT_COLORS,
    DEFAULT_MAX_SIZE_PX,
    ImageError,
    preprocess_image,
    resolve_image,
    svg_output_facts,
)
from .pipeline import Options, _flatten_on_white, idealize

# Transport is chosen at startup. stdio = local, full-trust (your own machine, your
# files). Any HTTP transport is potentially network-reachable, so the filesystem tools
# (arbitrary-path READ via idealize_logo, WRITE via output_path) are withheld and only
# the byte-in/SVG-out tools are exposed. See docs/mcp.md.
_TRANSPORT = (os.environ.get("VECTORMARK_MCP_TRANSPORT") or "stdio").strip()
_LOCAL_TRUST = _TRANSPORT == "stdio"

WIDGET_URI = "ui://vectormark/logo-widget.html"
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
        "output": {"svg_bytes": len(svg.encode()), **svg_output_facts(svg)},
        "warnings": warnings,
    }
    result = {
        "svg": svg,
        "width": meta.width,
        "height": meta.height,
        "svg_bytes": len(svg.encode()),
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
def idealize_logo(image: dict, options: dict | None = None) -> CallToolResult:
    """File-first logo idealization. Returns structured content plus a best-effort image block."""
    result, preview = idealize_logo_image(image, options, local_trust=_LOCAL_TRUST)
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
        "Idealize a base64-encoded raster image (e.g. one an agent just generated) into "
        "structured SVG, without needing a local file path. Accepts a bare base64 string "
        "or a data:image/...;base64,... URI."
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
    description=(
        "Render a vectormark SVG result in the ChatGPT/MCP Apps widget. "
        "Call idealize_logo first, then pass its returned fields here."
    ),
    meta={
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Rendering SVG preview...",
        "openai/toolInvocation/invoked": "Rendered SVG preview.",
    },
)
def render_idealized_logo(
    image_path: str,
    svg: str,
    width: int,
    height: int,
    output_path: str | None = None,
) -> dict[str, object]:
    """Render an existing idealized SVG result in the vectormark app."""

    return IdealizeLogoResult(
        image_path=image_path,
        output_path=output_path,
        width=width,
        height=height,
        svg_bytes=len(svg.encode()),   # derived from svg, not trusted from the caller
        svg=svg,
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
