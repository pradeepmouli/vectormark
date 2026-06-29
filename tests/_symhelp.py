"""Shared helper for symmetry regression tests: fold SSIM about a vertical axis."""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity
import resvg_py


def sym_iou_largest(svg: str, axis_x: float) -> float:
    """Rasterize `svg`, label connected foreground regions, and return the BEST
    pixel fold-SSIM (left half reflected onto right) among all regions that straddle
    `axis_x`, computed within each region's own y-extent.

    Using per-component fold SSIM isolates the symmetric body from asymmetric
    wordmarks or decorations that are spatially separate. The 0.999 threshold
    distinguishes truly symmetric vector shapes from near-symmetric rasters."""
    m = re.search(r'width="(\d+)".*?height="(\d+)"', svg)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (400, 400)

    png = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    arr = np.asarray(bg, dtype=np.uint8)

    fg = (arr < 255).any(axis=-1)   # True for any non-white pixel
    xi = int(round(axis_x))
    half = min(xi, w - xi)
    if half < 1:
        return 0.0

    labeled, n = ndi.label(fg)
    best_ssim = 0.0
    for i in range(1, n + 1):
        comp = labeled == i
        ys_idx, xs_idx = np.nonzero(comp)
        if xs_idx.size == 0 or xs_idx[0] >= xi or xs_idx[-1] <= xi:
            continue   # does not straddle axis_x
        y_min, y_max = int(ys_idx.min()), int(ys_idx.max()) + 1
        strip = arr[y_min:y_max]
        left = strip[:, xi - half:xi][:, ::-1]   # reflected left strip
        right = strip[:, xi:xi + half]
        ssim = float(structural_similarity(left, right, channel_axis=-1))
        if ssim > best_ssim:
            best_ssim = ssim

    return best_ssim
