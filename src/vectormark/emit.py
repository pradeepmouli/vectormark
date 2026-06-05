"""C6: serialise fitted shapes into a structured SVG document."""

from __future__ import annotations

import re

from .fit import Shape, _fmt
from .types import Axis

_KAPPA = 0.5522847498  # cubic-bézier circle constant
_PATH_TOKEN = re.compile(r"[MLCQZ]|-?\d*\.?\d+")
_COORD_COUNT = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}


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
        rule = f' fill-rule="{p["fill_rule"]}"' if p.get("fill_rule") else ""
        return f'<path {common}{rule} d="{p["d"]}"/>'
    raise ValueError(f"unknown shape kind: {shape.kind}")


def mirror_use(ref_id: str, axis: Axis) -> str:
    """Mirror element `ref_id` about the vertical axis via a reflection matrix."""
    return f'<use href="#{ref_id}" transform="matrix(-1 0 0 1 {_fmt(2 * axis.x)} 0)"/>'


def _ellipse_path_d(cx: float, cy: float, rx: float, ry: float) -> str:
    """Closed ellipse as four cubic-bézier quarter arcs (M/C only)."""
    ox, oy = rx * _KAPPA, ry * _KAPPA
    f = _fmt
    return (
        f"M{f(cx + rx)} {f(cy)} "
        f"C{f(cx + rx)} {f(cy + oy)} {f(cx + ox)} {f(cy + ry)} {f(cx)} {f(cy + ry)} "
        f"C{f(cx - ox)} {f(cy + ry)} {f(cx - rx)} {f(cy + oy)} {f(cx - rx)} {f(cy)} "
        f"C{f(cx - rx)} {f(cy - oy)} {f(cx - ox)} {f(cy - ry)} {f(cx)} {f(cy - ry)} "
        f"C{f(cx + ox)} {f(cy - ry)} {f(cx + rx)} {f(cy - oy)} {f(cx + rx)} {f(cy)} Z"
    )


def shape_to_path_d(shape: Shape) -> str:
    """Convert any fitted shape to a path `d` string using only M/L/C/Z, so a
    flattened SVG has no basic-shape elements and mirrors are a plain x-reflection."""
    p = shape.params
    if shape.kind == "path":
        return p["d"]
    if shape.kind == "circle":
        return _ellipse_path_d(p["cx"], p["cy"], p["r"], p["r"])
    if shape.kind == "ellipse":
        return _ellipse_path_d(p["cx"], p["cy"], p["rx"], p["ry"])
    if shape.kind == "rect":
        x, y, w, hh = p["x"], p["y"], p["w"], p["h"]
        f = _fmt
        return f"M{f(x)} {f(y)} L{f(x + w)} {f(y)} L{f(x + w)} {f(y + hh)} L{f(x)} {f(y + hh)} Z"
    if shape.kind == "polygon":
        pts = p["points"]
        f = _fmt
        body = " ".join(f"L{f(x)} {f(y)}" for x, y in pts[1:])
        return f"M{f(pts[0][0])} {f(pts[0][1])} {body} Z"
    raise ValueError(f"unknown shape kind: {shape.kind}")


def reflect_path_d(d: str, axis_x: float) -> str:
    """Reflect every x-coordinate of an absolute M/L/C/Z path about x = axis_x.
    (Our paths use only absolute M/L/C/Z, so a per-coordinate x flip is exact —
    no arc-sweep flags to invert.)"""
    toks = _PATH_TOKEN.findall(d)
    out: list[str] = []
    i = 0
    while i < len(toks):
        cmd = toks[i]
        out.append(cmd)
        i += 1
        n = _COORD_COUNT[cmd]
        coords = toks[i:i + n]
        i += n
        for j, c in enumerate(coords):
            v = 2 * axis_x - float(c) if j % 2 == 0 else float(c)
            out.append(_fmt(v))
    return " ".join(out)


def path_svg(d: str, fill: str, fill_rule: str | None = None) -> str:
    rule = f' fill-rule="{fill_rule}"' if fill_rule else ""
    return f'<path fill="{fill}"{rule} d="{d}"/>'


def render_svg_doc(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
