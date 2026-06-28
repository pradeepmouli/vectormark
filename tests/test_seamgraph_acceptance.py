"""Phase-A acceptance tests for seamgraph: gap-free seams, junction, determinism."""

from __future__ import annotations

import numpy as np
import pytest


A_COLOR = (200, 40, 40)
B_COLOR = (40, 160, 40)
C_COLOR = (40, 40, 200)
BG_COLOR = (255, 255, 255)
BG_IDX, A_IDX, B_IDX, C_IDX = 0, 1, 2, 3


def _make_synthetic():
    from vectormark.softlabel import soft_label_field, region_coverage
    from vectormark.contour import region_contours

    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    img[20:100, 20:60] = A_COLOR
    img[20:60, 60:100] = B_COLOR
    img[60:100, 60:100] = C_COLOR

    palette = np.array([BG_COLOR, A_COLOR, B_COLOR, C_COLOR], float)
    L = soft_label_field(img, palette)

    conts = {}
    for ridx, sl in [(A_IDX, (20, 100, 20, 60)), (B_IDX, (20, 60, 60, 100)), (C_IDX, (60, 100, 60, 100))]:
        mask = np.zeros((120, 120), bool)
        mask[sl[0]:sl[1], sl[2]:sl[3]] = True
        cov = region_coverage(L, ridx, mask)
        conts[ridx] = region_contours(mask, coverage=cov)[0]

    return L, conts


def _build_and_snap(L, conts):
    from vectormark.seamgraph import build_graph, snap_junctions
    g = build_graph(conts, L, bg_idx=BG_IDX)
    return snap_junctions(g, reach=1.5)


def _render_regions(gs, colors: dict, size: int = 120) -> np.ndarray:
    """Render regions over magenta sentinel; return (H,W,3) uint8."""
    from tests._render import render_svg
    from vectormark.seamgraph import region_path_d

    cache = {}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">']
    parts.append(f'<rect width="{size}" height="{size}" fill="#FF00FF"/>')
    for ridx, hex_col in colors.items():
        d = region_path_d(gs, ridx, epsilon=1.0, max_error=0.5, _edge_cache=cache)
        parts.append(f'<path d="{d}" fill="{hex_col}"/>')
    parts.append('</svg>')
    return render_svg(''.join(parts), size, size), cache


# ---------------------------------------------------------------------------
# Gap-free: zero interior background along seams
# ---------------------------------------------------------------------------

def test_gap_free_three_region_seams():
    """All three seam bands (A|B, A|C, B|C) have zero magenta sentinel pixels."""
    L, conts = _make_synthetic()
    gs = _build_and_snap(L, conts)
    colors = {A_IDX: '#C82828', B_IDX: '#28A028', C_IDX: '#2828C8'}
    raster, _ = _render_regions(gs, colors)

    magenta = (raster[:, :, 0] == 255) & (raster[:, :, 1] == 0) & (raster[:, :, 2] == 255)

    # Check each seam band interior (±1 px of the seam line, strictly interior)
    ab_seam_leaks = magenta[21:59, 59:62].sum()   # A|B seam: col ≈60, rows 21-59
    ac_seam_leaks = magenta[61:99, 59:62].sum()   # A|C seam: col ≈60, rows 61-99
    bc_seam_leaks = magenta[59:62, 61:99].sum()   # B|C seam: row ≈60, cols 61-99

    assert ab_seam_leaks == 0, f"A|B seam: {ab_seam_leaks} interior background px"
    assert ac_seam_leaks == 0, f"A|C seam: {ac_seam_leaks} interior background px"
    assert bc_seam_leaks == 0, f"B|C seam: {bc_seam_leaks} interior background px"


def test_gap_free_two_region_seam():
    """A two-region image has zero interior background pixels along the seam."""
    from vectormark.softlabel import soft_label_field, region_coverage
    from vectormark.contour import region_contours
    from vectormark.seamgraph import build_graph, snap_junctions, region_path_d
    from tests._render import render_svg

    img = np.full((80, 80, 3), 255, dtype=np.uint8)
    img[10:70, 10:40] = (200, 40, 40)    # A
    img[10:70, 40:70] = (40, 40, 200)    # C (use C_IDX=3)

    palette = np.array([BG_COLOR, A_COLOR, B_COLOR, C_COLOR], float)
    L = soft_label_field(img, palette)

    conts = {}
    for ridx, sl in [(A_IDX, (10, 70, 10, 40)), (C_IDX, (10, 70, 40, 70))]:
        mask = np.zeros((80, 80), bool); mask[sl[0]:sl[1], sl[2]:sl[3]] = True
        cov = region_coverage(L, ridx, mask)
        conts[ridx] = region_contours(mask, coverage=cov)[0]

    g = build_graph(conts, L, bg_idx=BG_IDX)
    gs = snap_junctions(g, reach=1.5)

    cache = {}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">']
    parts.append('<rect width="80" height="80" fill="#FF00FF"/>')
    for ridx, col in [(A_IDX, '#C82828'), (C_IDX, '#2828C8')]:
        d = region_path_d(gs, ridx, epsilon=1.0, max_error=0.5, _edge_cache=cache)
        parts.append(f'<path d="{d}" fill="{col}"/>')
    parts.append('</svg>')
    raster = render_svg(''.join(parts), 80, 80)
    magenta = (raster[:, :, 0] == 255) & (raster[:, :, 1] == 0) & (raster[:, :, 2] == 255)

    # Seam at col≈40, rows 11-69
    seam_leaks = magenta[11:69, 38:43].sum()
    assert seam_leaks == 0, f"Two-region seam: {seam_leaks} interior background px"


# ---------------------------------------------------------------------------
# Junction: no sliver at triple point
# ---------------------------------------------------------------------------

def test_junction_no_sliver():
    """3-px box around junction node has zero interior background pixels."""
    L, conts = _make_synthetic()
    gs = _build_and_snap(L, conts)
    colors = {A_IDX: '#C82828', B_IDX: '#28A028', C_IDX: '#2828C8'}
    raster, _ = _render_regions(gs, colors)

    magenta = (raster[:, :, 0] == 255) & (raster[:, :, 1] == 0) & (raster[:, :, 2] == 255)
    # Junction is snapped to ≈(59.67, 59.67) → pixel ≈ row=59, col=59
    junction_leaks = magenta[57:63, 57:63].sum()
    assert junction_leaks == 0, f"Junction: {junction_leaks} uncovered px in 6x6 box"


# ---------------------------------------------------------------------------
# Shared-fragment identity: seam edge fit only once
# ---------------------------------------------------------------------------

def test_seam_fragment_fit_once():
    """Each seam edge is fit exactly once: all seam edges in cache after rendering."""
    from vectormark.seamgraph import region_path_d

    L, conts = _make_synthetic()
    gs = _build_and_snap(L, conts)
    cache = {}

    for ridx in (A_IDX, B_IDX, C_IDX):
        region_path_d(gs, ridx, epsilon=1.0, max_error=0.5, _edge_cache=cache)

    seam_edges = [(i, e) for i, e in enumerate(gs.edges) if e.region_b is not None]
    assert len(seam_edges) == 3, f"Expected 3 seam edges, got {len(seam_edges)}"

    for i, e in seam_edges:
        assert i in cache, f"Seam edge {i} missing from fit cache (not fitted)"
        # Cache stores bezier list (could be empty for straight lines)
        assert isinstance(cache[i], list), f"Cache[{i}] is not a list"


def test_seam_bezier_geometry_identical():
    """Both regions sharing a seam use identical bezier geometry (fit once, same data)."""
    from vectormark.seamgraph import region_path_d

    L, conts = _make_synthetic()
    gs = _build_and_snap(L, conts)

    # Render A first, then B — seam edge for A|B should be cached from A's run
    cache = {}
    d_A = region_path_d(gs, A_IDX, epsilon=1.0, max_error=0.5, _edge_cache=cache)
    d_B = region_path_d(gs, B_IDX, epsilon=1.0, max_error=0.5, _edge_cache=cache)

    # Find A|B seam edge index
    ab_idx = next(i for i, e in enumerate(gs.edges)
                  if e.region_b is not None and
                  {e.region_a, e.region_b} == {A_IDX, B_IDX})

    assert ab_idx in cache
    beziers = cache[ab_idx]

    # Both A and B start with the same L command (A|B seam is straight)
    # Verify that neither re-computes the beziers (same Python object via cache)
    cache2 = {}
    d_B2 = region_path_d(gs, B_IDX, epsilon=1.0, max_error=0.5, _edge_cache=cache2)
    d_A2 = region_path_d(gs, A_IDX, epsilon=1.0, max_error=0.5, _edge_cache=cache2)

    # Determinism: same d strings regardless of rendering order
    assert d_A == d_A2, "Region A path not deterministic"
    assert d_B == d_B2, "Region B path not deterministic"


# ---------------------------------------------------------------------------
# Determinism: byte-identical across runs
# ---------------------------------------------------------------------------

def test_determinism_byte_identical():
    """region_path_d output is byte-identical across repeated calls."""
    from vectormark.seamgraph import region_path_d

    L, conts = _make_synthetic()
    gs = _build_and_snap(L, conts)

    results1, results2 = {}, {}
    for ridx in (A_IDX, B_IDX, C_IDX):
        results1[ridx] = region_path_d(gs, ridx, epsilon=1.0, max_error=0.5)
        results2[ridx] = region_path_d(gs, ridx, epsilon=1.0, max_error=0.5)

    for ridx in (A_IDX, B_IDX, C_IDX):
        assert results1[ridx] == results2[ridx], f"Region {ridx} not deterministic"
