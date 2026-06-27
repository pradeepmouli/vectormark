from __future__ import annotations

import asyncio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from PIL import Image, ImageDraw

from vectormark.mcp_server import WIDGET_URI, idealize_logo_file


def test_idealize_logo_file_returns_svg_and_writes_output(tmp_path):
    raster = tmp_path / "mark.png"
    output = tmp_path / "mark.svg"
    image = Image.new("RGB", (48, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 40, 40), fill=(30, 100, 220))
    image.save(raster)

    result = idealize_logo_file(str(raster), output_path=str(output), colors=4)

    assert result.image_path == str(raster.resolve())
    assert result.output_path == str(output.resolve())
    assert result.width == 48
    assert result.height == 48
    assert result.svg.startswith("<svg ")
    assert result.svg_bytes == len(result.svg.encode())
    assert output.read_text() == result.svg


def test_idealize_logo_file_reports_missing_input(tmp_path):
    missing = tmp_path / "missing.png"

    try:
        idealize_logo_file(str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing input should raise FileNotFoundError")


def test_stdio_server_exposes_idealize_logo_tool():
    import json

    async def list_tools_and_resources():
        params = StdioServerParameters(command="uv", args=["run", "vectormark-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                widget = await session.read_resource(WIDGET_URI)
                # file-first tool: pass image as a reference dict
                tool_result = await session.call_tool(
                    "idealize_logo",
                    {
                        "image": {"path": "tests/fixtures/daikonic/source.png"},
                        "options": {"colors": 4},
                    },
                )
                tool_meta = {
                    tool.name: tool.model_dump()
                    for tool in tools.tools
                }
                resource_uris = [str(resource.uri) for resource in resources.resources]
                widget_text = widget.contents[0].text
                # list return: dict→TextContent JSON, preview→ImageContent
                result_data = json.loads(tool_result.content[0].text)
                return tool_meta, resource_uris, widget_text, result_data

    tool_meta, resource_uris, widget_text, result = asyncio.run(list_tools_and_resources())

    assert "idealize_logo" in tool_meta
    meta = tool_meta["idealize_logo"]["meta"]
    assert meta["ui"]["resourceUri"] == WIDGET_URI
    assert meta["openai/fileParams"] == ["image"]
    assert "idealize_logo_data" in tool_meta
    assert "render_idealized_logo" in tool_meta
    assert WIDGET_URI in resource_uris
    assert "Logo idealizer" in widget_text
    assert "vectormark" in widget_text
    assert result["svg"].startswith("<svg ")
    assert result["diagnostics"]["input"]["source_kind"] == "local_path"


def _png_b64(width=60, height=60):
    import base64
    import io

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 50, 50), fill=(10, 30, 90))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_idealize_logo_bytes_returns_svg_and_writes_output(tmp_path):
    import base64

    from vectormark.mcp_server import idealize_logo_bytes

    raw = base64.b64decode(_png_b64(64, 48))
    output = tmp_path / "data.svg"
    result = idealize_logo_bytes(raw, output_path=str(output), colors=4)

    assert result.width == 64 and result.height == 48
    assert result.svg.startswith("<svg ")
    assert result.svg_bytes == len(result.svg.encode())
    assert result.image_path == "(inline base64 image)"   # no source path for inline data
    assert output.read_text() == result.svg


def test_decode_image_base64_accepts_data_uri_and_rejects_garbage():
    import base64

    import pytest

    from vectormark.mcp_server import _decode_image_base64

    b64 = _png_b64()
    raw = base64.b64decode(b64)
    assert _decode_image_base64(b64) == raw
    assert _decode_image_base64("data:image/png;base64," + b64) == raw   # data URI prefix stripped
    with pytest.raises(ValueError):
        _decode_image_base64("@@@ not base64 @@@")


def test_data_tool_ignores_output_path_in_http_mode(tmp_path, monkeypatch):
    # Under a network transport (_LOCAL_TRUST False) the data tool must NOT write to the
    # host filesystem even if a remote caller supplies output_path.
    import vectormark.mcp_server as srv
    monkeypatch.setattr(srv, "_LOCAL_TRUST", False)
    out = tmp_path / "should_not_exist.svg"
    result = srv.idealize_logo_data(_png_b64(), output_path=str(out))
    assert result["svg"].startswith("<svg ")
    assert result["output_path"] is None       # write suppressed
    assert not out.exists()                     # nothing written to disk


# ---------------------------------------------------------------------------
# Task 3: idealize_logo_image helper tests
# ---------------------------------------------------------------------------

import base64, io
import numpy as np
from PIL import Image as _Img


def _png_b64_solid(size=(48, 48), color=(30, 100, 220)):
    buf = io.BytesIO(); _Img.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_idealize_logo_image_from_data_uri():
    from vectormark.mcp_server import idealize_logo_image
    result, preview = idealize_logo_image(
        {"data_uri": f"data:image/png;base64,{_png_b64_solid()}"}, {"colors": 8}, local_trust=False)
    assert result["svg"].startswith("<svg ") and result["svg_bytes"] == len(result["svg"].encode())
    d = result["diagnostics"]
    assert d["input"]["source_kind"] == "data_uri" and d["input"]["mime_type"] == "image/png"
    assert d["vectormark"]["colors"] == 8 and "element_count" in d["output"]
    assert result["preview_available"] in (True, False)        # best-effort
    assert preview is None or isinstance(preview, (bytes, bytearray))


def test_idealize_logo_image_path_blocked_without_local_trust(tmp_path):
    from vectormark.mcp_server import idealize_logo_image
    from vectormark.mcp_image import ImageError
    p = tmp_path / "m.png"; _Img.new("RGB", (16, 16), "white").save(p)
    # local_trust=True works; False rejects
    r, _ = idealize_logo_image({"path": str(p)}, None, local_trust=True)
    assert r["svg"].startswith("<svg ")
    try:
        idealize_logo_image({"path": str(p)}, None, local_trust=False)
    except ImageError as e:
        assert e.error_code == "IMAGE_UNRESOLVABLE"
    else:
        raise AssertionError("path must be rejected without local trust")
