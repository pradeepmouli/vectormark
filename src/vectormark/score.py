"""Candidate scorer (prototype): structural-prior gates -> render-ΔE fidelity
gate -> structured description-length parsimony -> ΔE tiebreak. Ranks how an
element could be rendered. Standalone; not wired into the pipeline (slice 4)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import resvg_py
from PIL import Image

from .candidate import Candidate, FlatFill, LinearGradientFill, RadialGradientFill
from .color import mean_delta_e
from .emit import (
    linear_gradient_def, path_svg, radial_gradient_def, render_svg_doc, shape_to_path_d,
)
from .gradient import _BLOB_DOMINANCE, _MIN_STOP_SPAN, _dominant_blob_fraction, _stop_span
from .types import Region

# --- parsimony weights (tunable) -------------------------------------------------
_GEOM_PARAMS = {"circle": 3, "rect": 4, "ellipse": 5, "annulus": 6}
_CMD_COST = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}   # coords per path command
_FILL_FLAT = 1.0


def _path_cost(d: str) -> float:
    """Description length of a path `d`: sum of per-command coordinate counts."""
    return float(sum(_CMD_COST.get(cmd, 0) for cmd in re.findall(r"[MLCQZ]", d.upper())))


def parsimony_cost(cand: Candidate) -> float:
    """Structured description-length of a candidate: geometry params + fill params.
    Lower is simpler. Decoupled from SVG text formatting."""
    g = cand.geometry
    if g.kind == "polygon":
        geom = 2.0 * len(g.params["points"])
    elif g.kind == "path":
        geom = _path_cost(g.params["d"])
    else:
        geom = float(_GEOM_PARAMS[g.kind])

    f = cand.fill
    if isinstance(f, FlatFill):
        fill = _FILL_FLAT
    elif isinstance(f, LinearGradientFill):
        fill = 4.0 + 2.0 * len(f.stops)
    else:  # RadialGradientFill
        fill = 3.0 + 2.0 * len(f.stops)
    return geom + fill


# --- fidelity (render-ΔE via resvg) ---------------------------------------------
def _resolve_fill_str(fill, defs: list[str]) -> str:
    """Flat -> hex; gradient -> register a def (g{N}) and return url(#...)."""
    if isinstance(fill, FlatFill):
        return fill.hex
    g = fill.geometry
    gid = f"g{len(defs)}"
    if isinstance(fill, LinearGradientFill):
        defs.append(linear_gradient_def(gid, g["x1"], g["y1"], g["x2"], g["y2"], fill.stops))
    else:
        defs.append(radial_gradient_def(gid, g["cx"], g["cy"], g["r"], fill.stops))
    return f"url(#{gid})"


def _candidate_svg(cand: Candidate, w: int, h: int) -> str:
    defs: list[str] = []
    fill = _resolve_fill_str(cand.fill, defs)
    rule = cand.geometry.params.get("fill_rule",
                                    "evenodd" if cand.geometry.kind == "annulus" else None)
    body = [path_svg(shape_to_path_d(cand.geometry), fill, rule)]
    return render_svg_doc(w, h, body, defs)


def _rasterize(svg: str, w: int, h: int) -> np.ndarray:
    """SVG -> (h, w, 3) uint8 composited on white. (Mirrors tests/_render.render_svg;
    kept local so src/ does not import test helpers — unify in a later DRY pass.)"""
    png = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    return np.asarray(bg, dtype=np.uint8)


def render_delta_e(cand: Candidate, source_rgb: np.ndarray, region: Region) -> float:
    """Render the candidate over the source canvas and compare (mean OKLab ΔE)
    against the source within the region's footprint. 0 = identical."""
    h, w = source_rgb.shape[:2]
    mask = region.mask
    if not mask.any():
        return float("inf")
    raster = _rasterize(_candidate_svg(cand, w, h), w, h)
    return mean_delta_e(source_rgb[mask], raster[mask])


# --- structural priors (hard gates) ---------------------------------------------
def structural_priors(cand: Candidate, region: Region) -> tuple[bool, str | None]:
    """Hard gates generalising the proven gradient guards. A failure disqualifies
    the candidate (with a reason); flat/primitive/path candidates have no prior."""
    f = cand.fill
    if isinstance(f, (LinearGradientFill, RadialGradientFill)):
        if _stop_span(f.stops) < _MIN_STOP_SPAN:
            return False, "gradient stop-span below minimum"
        if _dominant_blob_fraction(region.mask) < _BLOB_DOMINANCE:
            return False, "gradient footprint not a single dominant blob"
    return True, None
