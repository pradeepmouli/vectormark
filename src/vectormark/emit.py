"""C6: serialise fitted shapes into a structured SVG document."""

from __future__ import annotations

import math
import re

from .fit import Shape, _fmt
from .types import Axis

_KAPPA = 0.5522847498  # cubic-bézier circle constant
_PATH_TOKEN = re.compile(r"[MLCQZ]|-?\d*\.?\d+")
_COORD_COUNT = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}
_AFFINE_TOKEN = re.compile(r"[MLCQAZ]|-?\d*\.?\d+")


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
    if shape.kind == "annulus":
        d = _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
        return f'<path {common} fill-rule="evenodd" d="{d}"/>'
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


def _annulus_path_d(cx: float, cy: float, r_outer: float, r_inner: float) -> str:
    """Two concentric circle subpaths (outer + inner); under even-odd fill the
    inner subpath cuts the hole."""
    return _ellipse_path_d(cx, cy, r_outer, r_outer) + " " + _ellipse_path_d(cx, cy, r_inner, r_inner)


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
    if shape.kind == "annulus":
        return _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
    raise ValueError(f"unknown shape kind: {shape.kind}")


def reflect_path_d(d: str, axis_x: float) -> str:
    """Reflect every x-coordinate of an absolute M/L/C/Q/Z path about x = axis_x.

    Assumes one explicit command per coordinate set (no implicit-repeat tokens like
    ``M0 0 10 0``) and no arc ``A`` flags to invert — which is exactly what this
    package's emitters produce. A per-coordinate x flip is then exact."""
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


def apply_affine_point(bake: tuple[float, float, float, float, float, float],
                       x: float, y: float) -> tuple[float, float]:
    """Apply an SVG affine (a, b, c, d, e, f) to a point: (a*x + c*y + e, b*x + d*y + f)."""
    a, b, c, d, e, f = bake
    return (a * x + c * y + e, b * x + d * y + f)


def transform_path_d(d: str, m: tuple[float, float, float, float, float, float]) -> str:
    """Apply the SVG affine `m = (a, b, c, d, e, f)` to every point of an absolute
    path, so the transform can be *baked* into the geometry instead of carried on a
    wrapping `<g transform>`. Handles M/L/C/Q/Z exactly; for an elliptical-arc `A`
    only the endpoint moves and the x-axis-rotation advances by the affine's
    rotation — valid because the transforms we bake are rigid (rotation +
    translation), which leave `rx`, `ry`, and the arc flags invariant."""
    a, b, c, dd, e, f = m
    rot_deg = math.degrees(math.atan2(b, a))

    def pt(x: float, y: float) -> tuple[float, float]:
        return apply_affine_point(m, x, y)

    toks = _AFFINE_TOKEN.findall(d)
    out: list[str] = []
    i = 0
    while i < len(toks):
        cmd = toks[i]
        out.append(cmd)
        i += 1
        if cmd == "Z":
            continue
        if cmd == "A":
            rx, ry, xrot, large, sweep, x, y = toks[i:i + 7]
            i += 7
            nx, ny = pt(float(x), float(y))
            out += [rx, ry, _fmt(float(xrot) + rot_deg), large, sweep, _fmt(nx), _fmt(ny)]
            continue
        n = _COORD_COUNT[cmd]
        nums = [float(t) for t in toks[i:i + n]]
        i += n
        for k in range(0, n, 2):
            nx, ny = pt(nums[k], nums[k + 1])
            out += [_fmt(nx), _fmt(ny)]
    return " ".join(out)


def path_svg(d: str, fill: str, fill_rule: str | None = None) -> str:
    rule = f' fill-rule="{fill_rule}"' if fill_rule else ""
    return f'<path fill="{fill}"{rule} d="{d}"/>'


def _gradient_stops(stops: list[tuple[float, str]]) -> str:
    return "".join(f'<stop offset="{_fmt(o)}" stop-color="{c}"/>' for o, c in stops)


def linear_gradient_def(elem_id: str, x1: float, y1: float, x2: float, y2: float,
                        stops: list[tuple[float, str]]) -> str:
    """A <linearGradient> in userSpaceOnUse coords (absolute px; no gradientTransform,
    survives --flatten)."""
    return (f'<linearGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">'
            f'{_gradient_stops(stops)}</linearGradient>')


def radial_gradient_def(elem_id: str, cx: float, cy: float, r: float,
                        stops: list[tuple[float, str]]) -> str:
    """A <radialGradient> in userSpaceOnUse coords."""
    return (f'<radialGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}">'
            f'{_gradient_stops(stops)}</radialGradient>')


def render_svg_doc(width: int, height: int, body: list[str], defs: list[str] | None = None) -> str:
    defs_block = f'  <defs>{"".join(defs)}</defs>\n  ' if defs else "  "
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        + defs_block
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
