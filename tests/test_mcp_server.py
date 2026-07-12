from __future__ import annotations

import asyncio
from typing import get_args, get_type_hints
import base64
import io
from types import SimpleNamespace

import pytest
import vectormark.mcp_server as mcp_server_module
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp.exceptions import ToolError
from PIL import Image, ImageDraw

from vectormark.mcp_server import (
    WIDGET_URI,
    ImageRef,
    TraceDrawingOptions,
    get_drawing_artifact,
    mcp,
    refine_drawing,
    render_drawing,
    trace_drawing,
)


def _png_data_uri() -> str:
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill=(20, 100, 240))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def test_interactive_trace_refine_and_artifact_share_one_live_region_forest():
    """The public MCP helpers preserve v0 roots and refine child versions."""
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx)
    trace = traced.structuredContent
    assert trace is not None
    drawing_id = trace["drawing_id"]
    assert trace["version"] == "v0"
    assert trace["report"]["targets"]
    assert "region_map_svg" not in trace["trace"]
    assert "geometry" in trace["trace"]["regions"][0]
    raw_trace = get_drawing_artifact(drawing_id, "v0", "raw_trace", ctx).structuredContent
    assert raw_trace is not None
    assert "trace_path" in raw_trace["trace"]["regions"][0]

    refined = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": drawing_id,
            "base_version": "v0",
            "ops": [{"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}}],
        },
        ctx,
    )
    result = refined.structuredContent
    assert result is not None and result["version"] == "v0.0"
    assert result["report"]["targets"][0]["geometry"] == "circle"

    artifact = get_drawing_artifact(drawing_id, "v0.0", "svg", ctx).structuredContent
    assert artifact is not None and '<circle id="r1"' in artifact["svg"]


def test_interactive_trace_stitches_retained_roots_before_creating_v0(monkeypatch):
    observed: list[int] = []
    original_stitch = mcp_server_module.stitch_regions

    def observe(trace, regions):
        observed.append(len(regions))
        return original_stitch(trace, regions)

    monkeypatch.setattr(mcp_server_module, "stitch_regions", observe)
    ctx = SimpleNamespace(session=object())

    result = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structuredContent

    assert result is not None
    assert observed == [len(result["trace"]["regions"])]


def test_mcp_detect_symmetry_exposes_an_axis_for_a_follow_up_set_symmetry_plan():
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structuredContent
    assert traced is not None
    drawing_id = traced["drawing_id"]

    detected = refine_drawing(
        {
            "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": "v0",
            "ops": [{"op": "detect_symmetry", "target": "r1", "epsilon": 0.5}],
        },
        ctx,
    ).structuredContent
    assert detected is not None
    children = [target for target in detected["report"]["targets"] if target["id"].startswith("r1-")]
    source = next(target for target in children if target["diagnostics"].get("symmetry", {}).get("mode") == "self")
    target = next(target for target in children if target["id"] != source["id"])

    set_result = refine_drawing(
        {
            "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": detected["version"],
            "ops": [{"op": "set_symmetry", "source": source["id"], "target": target["id"],
                     "axis": source["diagnostics"]["symmetry"]["axis"]}],
        },
        ctx,
    ).structuredContent
    assert set_result is not None
    updated = next(item for item in set_result["report"]["targets"] if item["id"] == target["id"])
    assert updated["diagnostics"]["symmetry"]["mode"] == "pair"


def test_trace_auto_returns_a_one_shot_svg_without_live_state():
    ctx = SimpleNamespace(session=object())
    result = trace_drawing(
        ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="auto", max_colors=4), ctx
    ).structuredContent

    assert result is not None
    assert result["svg"].startswith("<svg ")
    assert "drawing_id" not in result


def test_render_drawing_accepts_widget_result_shape():
    result = {"svg": "<svg>x</svg>", "width": 10, "height": 12}

    rendered = render_drawing(result=result)

    assert rendered["svg"] == result["svg"]
    assert rendered["svg_bytes"] == len(result["svg"].encode())


def test_trace_drawing_surfaces_image_error_codes():
    async def call():
        return await mcp.call_tool(
            "trace_drawing",
            {"image": {"data_uri": "data:image/png;base64,AA=="}, "options": {"refine": "auto"}},
        )

    try:
        asyncio.run(call())
        raise AssertionError("Expected trace_drawing to reject invalid image data")
    except ToolError as exc:
        assert "UNSUPPORTED_IMAGE_TYPE" in str(exc)


def test_refine_drawing_converts_an_executor_error_into_a_plan_error(monkeypatch):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structuredContent
    assert traced is not None

    def fail(*_args, **_kwargs):
        raise ValueError("executor rejected this valid-looking plan")

    monkeypatch.setattr(mcp_server_module, "refine", fail)
    with pytest.raises(ToolError, match="PLAN_INVALID.*executor rejected"):
        refine_drawing(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": traced["drawing_id"],
                "base_version": "v0",
                "ops": [{"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}}],
            },
            ctx,
        )


def test_stdio_server_exposes_only_the_drawing_first_surface():
    async def list_tools_and_run_drawing_workflow():
        params = StdioServerParameters(command="uv", args=["run", "vectormark-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                resources = await session.list_resources()
                widget = await session.read_resource(WIDGET_URI)
                auto = await session.call_tool(
                    "trace_drawing",
                    {"image": {"data_uri": _png_data_uri()}, "options": {"refine": "auto", "max_colors": 4}},
                )
                traced = await session.call_tool(
                    "trace_drawing",
                    {"image": {"data_uri": _png_data_uri()}, "options": {"refine": "interactive", "max_colors": 4}},
                )
                drawing_id = traced.structuredContent["drawing_id"]
                refined = await session.call_tool(
                    "refine_drawing",
                    {
                        "plan": {
                            "version": "vectormark.plan.v1",
                            "drawing_id": drawing_id,
                            "base_version": "v0",
                            "ops": [{"op": "set_geometry", "target": "r1", "geometry": {"type": "circle"}}],
                        }
                    },
                )
                artifact = await session.call_tool(
                    "get_drawing_artifact",
                    {"drawing_id": drawing_id, "version": refined.structuredContent["version"], "artifact": "svg"},
                )
                return initialized, tools, [str(resource.uri) for resource in resources.resources], widget.contents[0].text, auto, refined, artifact

    initialized, tools, resources, widget, auto, refined, artifact = asyncio.run(list_tools_and_run_drawing_workflow())

    assert set(tools) == {"trace_drawing", "refine_drawing", "get_drawing_artifact", "render_drawing"}
    assert "interactive" in (initialized.instructions or "")
    assert tools["trace_drawing"].meta["openai/fileParams"] == ["image"]
    refine_schema = tools["refine_drawing"].inputSchema
    plan_schema = refine_schema["$defs"]["RefinementPlanInput"]
    assert "oneOf" in plan_schema["properties"]["ops"]["items"]
    assert "set_geometry" in str(refine_schema)
    assert "r1.p1.c14-1" in str(refine_schema)
    assert "match_length" in str(refine_schema)
    assert "quadratic" in str(refine_schema)
    for tool_name in ("trace_drawing", "refine_drawing", "get_drawing_artifact", "render_drawing"):
        assert tools[tool_name].outputSchema is not None
    assert WIDGET_URI in resources
    assert "Logo idealizer" in widget
    assert auto.structuredContent is not None
    assert auto.structuredContent["svg"].startswith("<svg ")
    assert refined.structuredContent["version"] == "v0.0"
    assert '<circle id="r1"' in artifact.structuredContent["svg"]


def test_drawing_artifact_accepts_the_preview_alias_returned_by_trace_and_refine():
    from vectormark.mcp_server import get_drawing_artifact

    artifact_type = get_type_hints(get_drawing_artifact)["artifact"]

    assert "preview" in get_args(artifact_type)
