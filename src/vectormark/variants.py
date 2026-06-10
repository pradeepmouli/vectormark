# src/vectormark/variants.py
"""Whole-mark variant matrix: idealize one raster across an epsilon × max_error
grid, for SVG export, a JSON manifest, and an annotated contact sheet."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .fit import _fmt
from .pipeline import IdealizeReport, Options, _flatten_on_white, idealize

DEFAULT_EPSILONS = (0.5, 1.5, 3.0)
DEFAULT_MAX_ERRORS = (0.5, 1.0, 2.5)


@dataclass(frozen=True)
class Variant:
    """One matrix cell: the (epsilon, max_error) used, the emitted SVG, the
    strategy report, and an error string if that cell failed to idealize."""

    epsilon: float
    max_error: float
    svg: str
    report: IdealizeReport
    error: str | None = None


def _as_rgb(image) -> np.ndarray:
    """Load any accepted image input to an (H, W, 3) uint8 array ONCE, so the grid
    does not re-read/re-flatten the source per cell."""
    if isinstance(image, str):
        with Image.open(image) as im:
            return _flatten_on_white(im)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _flatten_on_white(Image.fromarray(arr, "RGBA"))
    return arr


def _axis_tag(value: float) -> str:
    """Filename-safe axis token: 0.5 -> '0_5', 3.0 -> '3', 1.0 -> '1'."""
    return _fmt(value).replace(".", "_")


def variant_filename(v: Variant) -> str:
    return f"variant-e{_axis_tag(v.epsilon)}-m{_axis_tag(v.max_error)}.svg"


def write_variant_set(variants: list[Variant], out_dir, *, source: str) -> Path:
    """Write one SVG per successful cell plus manifest.json into out_dir (created if
    needed). A failed cell contributes a manifest entry with `error` and no `file`.
    Returns the out_dir path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    epsilons, max_errors = [], []
    for v in variants:
        if v.epsilon not in epsilons:
            epsilons.append(v.epsilon)
        if v.max_error not in max_errors:
            max_errors.append(v.max_error)

    entries = []
    for v in variants:
        entry = {"epsilon": v.epsilon, "max_error": v.max_error}
        if v.error is not None:
            entry["error"] = v.error
        else:
            fname = variant_filename(v)
            (out / fname).write_text(v.svg)
            entry.update(
                file=fname,
                svg_bytes=len(v.svg.encode()),
                strategies=dict(v.report.strategies),
                gradients=v.report.gradients,
                elements=v.report.elements,
            )
        entries.append(entry)

    manifest = {
        "source": source,
        "axes": {"epsilon": epsilons, "max_error": max_errors},
        "variants": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def generate_variants(
    image, *, epsilons=DEFAULT_EPSILONS, max_errors=DEFAULT_MAX_ERRORS,
    base: Options | None = None,
) -> list[Variant]:
    """idealize() the image once per (epsilon, max_error) cell, row-major (epsilon
    outer, max_error inner). `base` supplies the non-geometry knobs held constant
    across the grid (max_colors, no_symmetry, …); epsilon/max_error are overridden
    per cell. A cell that raises is captured as a failed Variant, never aborting
    the grid."""
    arr = _as_rgb(image)
    base = base or Options()
    out: list[Variant] = []
    for eps in epsilons:
        for me in max_errors:
            opt = replace(base, epsilon=eps, max_error=me)
            try:
                svg, report = idealize(arr, options=opt, report=True)
                out.append(Variant(eps, me, svg, report))
            except Exception as exc:                       # one bad cell must not kill the grid
                out.append(Variant(eps, me, "", IdealizeReport.empty(), error=str(exc)))
    return out
