"""C6: serialise fitted shapes into a structured SVG document."""

from __future__ import annotations

from .fit import Shape, _fmt
from .types import Axis


def shape_to_svg(shape: Shape, fill: str, elem_id: str) -> str:
    p = shape.params
    common = f'id="{elem_id}" fill="{fill}"'
    if shape.kind == "circle":
        return f'<circle {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" r="{_fmt(p["r"])}"/>'
    if shape.kind == "ellipse":
        return (f'<ellipse {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" '
                f'rx="{_fmt(p["rx"])}" ry="{_fmt(p["ry"])}"/>')
    if shape.kind == "rect":
        return (f'<rect {common} x="{_fmt(p["x"])}" y="{_fmt(p["y"])}" '
                f'width="{_fmt(p["w"])}" height="{_fmt(p["h"])}"/>')
    if shape.kind == "polygon":
        pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p["points"])
        return f'<polygon {common} points="{pts}"/>'
    if shape.kind == "path":
        return f'<path {common} d="{p["d"]}"/>'
    raise ValueError(f"unknown shape kind: {shape.kind}")


def mirror_use(ref_id: str, axis: Axis) -> str:
    """Mirror element `ref_id` about the vertical axis via a reflection matrix."""
    return f'<use href="#{ref_id}" transform="matrix(-1 0 0 1 {_fmt(2 * axis.x)} 0)"/>'


def render_svg_doc(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
