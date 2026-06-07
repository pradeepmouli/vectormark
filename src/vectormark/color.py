"""A0 colour optimisation: OKLab conversion, ΔE, palette extraction, quantise."""

from __future__ import annotations

import numpy as np

# sRGB <-> linear
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# OKLab matrices (Björn Ottosson, 2020)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) in [0,1] -> OKLab (..., 3)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    lms = lin @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> sRGB (..., 3) clipped to [0,1]."""
    lms_ = np.asarray(lab, dtype=float) @ _M2_INV.T
    lms = lms_ ** 3
    lin = lms @ _M1_INV.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """OKLab Euclidean distance (OKLab is designed for L2 ΔE)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def extract_palette(
    rgb_image: np.ndarray, *, max_colors: int = 16, merge_de: float = 0.045,
    min_fraction: float = 0.002,
) -> np.ndarray:
    """Perceptual-clustering palette in OKLab.

    Clusters near-identical shades *before* applying the frequency floor, so a
    colour dispersed across many antialiased shades (a thin/small mark) is kept
    by its aggregate weight instead of dropped per-shade. Each cluster's
    representative is its most-frequent member — a real colour, never a centroid.

    Returns an (N, 3) uint8 array, frequency-ordered.
    """
    flat = np.asarray(rgb_image, dtype=np.uint8).reshape(-1, 3)
    total = len(flat)
    if total == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Distinct full-precision colours and their pixel counts.
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    # Coarse 5-bit pre-bin (32 levels/channel) caps clustering input regardless
    # of AA spread. Group full-precision colours by their bin.
    color_bin = colors >> 3
    bins, bin_inv = np.unique(color_bin, axis=0, return_inverse=True)
    bin_inv = bin_inv.ravel()
    nbins = len(bins)
    bin_total = np.zeros(nbins, dtype=np.int64)
    np.add.at(bin_total, bin_inv, counts)

    # Representative per bin = most-frequent full-precision colour in that bin.
    # Order colours by descending count, value-tiebroken for determinism; the
    # first colour seen for each bin (in this order) is its representative.
    order = np.lexsort((colors[:, 2], colors[:, 1], colors[:, 0], -counts))
    sorted_bins = bin_inv[order]
    uniq_bin, first_pos = np.unique(sorted_bins, return_index=True)
    rep_color_idx = np.empty(nbins, dtype=np.int64)
    rep_color_idx[uniq_bin] = order[first_pos]
    rep_color = colors[rep_color_idx]                       # (nbins, 3) full precision
    rep_lab = srgb_to_oklab(rep_color / 255.0)

    # Greedy perceptual clustering over bins, in descending aggregate-count
    # order (stable -> value-tiebroken via the value-sorted bin ids). The seed
    # bin of each cluster is its most-frequent member -> the representative.
    # A new cluster is only seeded by a bin whose own total meets the floor;
    # sub-floor bins can only be absorbed into an existing cluster (AA shades).
    # Trade-off: a colour whose *most-frequent* bin is still sub-floor (e.g. a
    # hue smeared over many distant bins) is dropped — it has no dominant
    # representative to anchor a cluster. Safe direction: a miss, not a false
    # colour. See the spec's "Known limitation (from the seed-floor rule)".
    b_order = np.argsort(-bin_total, kind="stable")
    floor = min_fraction * total
    cluster_lab: list[np.ndarray] = []
    cluster_total: list[int] = []
    cluster_bin: list[int] = []
    for b in b_order:
        placed = False
        for k in range(len(cluster_lab)):
            if delta_e(rep_lab[b], cluster_lab[k]) < merge_de:
                cluster_total[k] += int(bin_total[b])
                placed = True
                break
        if not placed and bin_total[b] >= floor:
            cluster_lab.append(rep_lab[b])
            cluster_total.append(int(bin_total[b]))
            cluster_bin.append(int(b))

    # Cap to max_colors, frequency-ordered.
    if not cluster_total:
        # All bins are sub-floor: fall back to the most-frequent bin.
        fb = int(b_order[0])
        return rep_color[fb:fb + 1].astype(np.uint8)
    totals = np.array(cluster_total, dtype=np.int64)
    keep = list(np.argsort(-totals, kind="stable")[:max_colors])
    return rep_color[[cluster_bin[k] for k in keep]].astype(np.uint8)


def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel OKLab Euclidean distance between two uint8 RGB images/arrays
    (shape (...,3)). 0.0 == identical; perceptual color error."""
    la = srgb_to_oklab(a.reshape(-1, 3) / 255.0)
    lb = srgb_to_oklab(b.reshape(-1, 3) / 255.0)
    return float(np.linalg.norm(la - lb, axis=1).mean())


def quantize(rgb_image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Assign every pixel to its nearest palette colour by OKLab ΔE."""
    h, w, _ = rgb_image.shape
    flat = srgb_to_oklab(rgb_image.reshape(-1, 3) / 255.0)
    pal_lab = srgb_to_oklab(palette / 255.0)
    # pairwise distances (P pixels x K palette)
    d = np.linalg.norm(flat[:, None, :] - pal_lab[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return palette[nearest].reshape(h, w, 3).astype(np.uint8)
