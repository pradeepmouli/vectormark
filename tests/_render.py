"""Shared test helpers: render an SVG to RGB and compare to a reference."""

from __future__ import annotations

import io

import numpy as np
import resvg_py
from PIL import Image
from skimage.metrics import structural_similarity

from vectormark.color import mean_delta_e  # re-exported; single source of truth


def render_svg(svg: str, width: int, height: int) -> np.ndarray:
    """Rasterize `svg` to an (H, W, 3) uint8 array on a white background."""
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    return np.asarray(bg, dtype=np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity in [0, 1]; 1.0 == identical."""
    return float(structural_similarity(a, b, channel_axis=-1))
