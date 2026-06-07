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
