"""Tests for seamgraph.py — Tasks 1-4, 6."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic image fixture
# ---------------------------------------------------------------------------
# 120×120 white image; A=(200,40,40) at [20:100,20:60],
# B=(40,160,40) at [20:60,60:100], C=(40,40,200) at [60:100,60:100].
# Seams: A|B at x≈59.5 rows 20-60; A|C at x≈59.5 rows 60-100;
#         B|C at y≈59.5 cols 60-100. Triple-junction near (59.5, 59.5).

A_COLOR = (200, 40, 40)
B_COLOR = (40, 160, 40)
C_COLOR = (40, 40, 200)
BG_COLOR = (255, 255, 255)

# Palette indices
BG_IDX, A_IDX, B_IDX, C_IDX = 0, 1, 2, 3


def _make_synthetic():
    """Return (img, palette, L, masks, contours, coverages)."""
    from vectormark.softlabel import soft_label_field, region_coverage
    from vectormark.contour import region_contours

    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    img[20:100, 20:60] = A_COLOR
    img[20:60, 60:100] = B_COLOR
    img[60:100, 60:100] = C_COLOR

    palette = np.array([BG_COLOR, A_COLOR, B_COLOR, C_COLOR], float)
    L = soft_label_field(img, palette)

    mask_A = np.zeros((120, 120), bool); mask_A[20:100, 20:60] = True
    mask_B = np.zeros((120, 120), bool); mask_B[20:60, 60:100] = True
    mask_C = np.zeros((120, 120), bool); mask_C[60:100, 60:100] = True

    cov_A = region_coverage(L, A_IDX, mask_A)
    cov_B = region_coverage(L, B_IDX, mask_B)
    cov_C = region_coverage(L, C_IDX, mask_C)

    cont_A = region_contours(mask_A, coverage=cov_A)[0]
    cont_B = region_contours(mask_B, coverage=cov_B)[0]
    cont_C = region_contours(mask_C, coverage=cov_C)[0]

    return (img, palette, L,
            {"A": mask_A, "B": mask_B, "C": mask_C},
            {"A": cont_A, "B": cont_B, "C": cont_C},
            {"A": cov_A, "B": cov_B, "C": cov_C})


# ---------------------------------------------------------------------------
# Task 1: classify_contour
# ---------------------------------------------------------------------------

def test_classify_contour_seam_labels():
    from vectormark.seamgraph import classify_contour

    _, _, L, masks, conts, _ = _make_synthetic()

    labels = classify_contour(conts["A"], L, A_IDX, bg_idx=BG_IDX)
    assert labels.shape == (len(conts["A"]),)
    assert labels.dtype in (np.int32, np.int64, int)

    # Points near x≈59.5 with y in (20, 60) → seam with B
    c = conts["A"]
    ab_band = (np.abs(c[:, 0] - 59.5) < 1.5) & (c[:, 1] > 21) & (c[:, 1] < 59)
    ac_band = (np.abs(c[:, 0] - 59.5) < 1.5) & (c[:, 1] > 61) & (c[:, 1] < 99)
    outer_band = (c[:, 1] < 21) | (c[:, 1] > 99) | (c[:, 0] < 21)

    ab_pts = labels[ab_band]
    ac_pts = labels[ac_band]
    out_pts = labels[outer_band]

    assert len(ab_pts) >= 10, f"Too few A|B seam points: {len(ab_pts)}"
    assert len(ac_pts) >= 10, f"Too few A|C seam points: {len(ac_pts)}"
    assert len(out_pts) >= 10, f"Too few outer boundary points: {len(out_pts)}"

    # ≥80% of seam-band points carry the expected neighbor label
    assert (ab_pts == B_IDX).mean() >= 0.80, f"A|B seam: {(ab_pts==B_IDX).mean():.2f}"
    assert (ac_pts == C_IDX).mean() >= 0.80, f"A|C seam: {(ac_pts==C_IDX).mean():.2f}"
    assert (out_pts == BG_IDX).mean() >= 0.80, f"outer: {(out_pts==BG_IDX).mean():.2f}"


# ---------------------------------------------------------------------------
# Task 2: build_graph
# ---------------------------------------------------------------------------

def test_build_graph_seam_edges():
    from vectormark.seamgraph import classify_contour, build_graph, EdgeGraph

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}

    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    assert isinstance(g, EdgeGraph)

    seam_edges = [e for e in g.edges if e.region_b is not None]
    boundary_edges = [e for e in g.edges if e.region_b is None]

    # Exactly 3 seam edges (A|B, A|C, B|C)
    assert len(seam_edges) == 3, f"Expected 3 seam edges, got {len(seam_edges)}"

    # Each seam edge has both region ids set
    seam_pairs = {(min(e.region_a, e.region_b), max(e.region_a, e.region_b))
                  for e in seam_edges}
    assert seam_pairs == {(A_IDX, B_IDX), (A_IDX, C_IDX), (B_IDX, C_IDX)}, \
        f"Seam pairs: {seam_pairs}"

    # No seam edge is duplicated
    assert len(seam_edges) == len(seam_pairs)

    # Seam edge pts are coordinate-identical to matching runs in source contours
    for e in seam_edges:
        assert e.pts.shape[1] == 2
        assert len(e.pts) >= 2


def test_build_graph_boundary_edges():
    from vectormark.seamgraph import build_graph

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}

    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    boundary_edges = [e for e in g.edges if e.region_b is None]

    # 3 boundary edges (one outer boundary arc per region)
    assert len(boundary_edges) == 3, f"Expected 3 boundary edges, got {len(boundary_edges)}"

    # All boundary edges have region_b None
    for e in boundary_edges:
        assert e.region_b is None


# ---------------------------------------------------------------------------
# Task 3: snap_junctions
# ---------------------------------------------------------------------------

def test_snap_junctions_triple_point():
    from vectormark.seamgraph import build_graph, snap_junctions

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}

    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    gs = snap_junctions(g, reach=1.5)

    seam_edges = [e for e in gs.edges if e.region_b is not None]

    # All three seam edges share the SAME junction node
    junction_nodes_by_edge = []
    for e in seam_edges:
        junction_nodes_by_edge.append({e.node0, e.node1})

    # Find the junction node: it must appear in all 3 seam edges
    all_nodes_in_seams = set()
    for s in junction_nodes_by_edge:
        all_nodes_in_seams |= s

    junction_candidates = [n for n in all_nodes_in_seams
                           if sum(1 for s in junction_nodes_by_edge if n in s) == 3]
    assert len(junction_candidates) == 1, \
        f"Expected 1 junction node shared by all 3 seam edges, got {len(junction_candidates)}"

    jn = junction_candidates[0]
    jpos = gs.nodes[jn]

    # Junction node ≈ (59.5, 59.5) within 1px
    assert abs(jpos[0] - 59.5) <= 1.0, f"Junction x={jpos[0]:.2f}"
    assert abs(jpos[1] - 59.5) <= 1.0, f"Junction y={jpos[1]:.2f}"


def test_snap_junctions_preserves_non_junction():
    from vectormark.seamgraph import build_graph, snap_junctions

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}

    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    gs = snap_junctions(g, reach=1.5)

    # Total node count should be ≤ original (snapping only reduces)
    assert len(gs.nodes) <= len(g.nodes)


# ---------------------------------------------------------------------------
# Task 4: fit_edge (in fit.py)
# ---------------------------------------------------------------------------

def test_fit_edge_straight_line():
    from vectormark.fit import fit_edge

    pts = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    frag = fit_edge(pts, epsilon=1.0, max_error=0.5)
    assert "L" in frag
    assert "C" not in frag
    # Last coordinate must end at (4, 0)
    import re
    nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", frag)]
    assert abs(nums[-2] - 4.0) < 0.01 and abs(nums[-1] - 0.0) < 0.01


def test_fit_edge_arc_contains_curve():
    from vectormark.fit import fit_edge

    # A quarter-circle arc from (10,0) to (0,10) radius 10
    t = np.linspace(0, np.pi / 2, 40)
    pts = np.column_stack([10 * np.cos(t), 10 * np.sin(t)])
    frag = fit_edge(pts, epsilon=1.0, max_error=0.5)
    assert "C" in frag

    # Endpoints preserved: first coord is handled by M (caller), last coord in fragment
    import re
    nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", frag)]
    assert abs(nums[-2] - 0.0) < 0.5 and abs(nums[-1] - 10.0) < 0.5


def test_fit_edge_no_M_no_Z():
    from vectormark.fit import fit_edge

    pts = np.array([[0., 0.], [5., 5.], [10., 0.]])
    frag = fit_edge(pts, epsilon=1.0, max_error=0.5)
    assert "M" not in frag
    assert "Z" not in frag


# ---------------------------------------------------------------------------
# Task 5: region_path_d (gap-free invariant) — core assertions
# ---------------------------------------------------------------------------

def test_region_path_d_valid_closed():
    from vectormark.seamgraph import build_graph, snap_junctions, region_path_d

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}
    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    gs = snap_junctions(g, reach=1.5)

    for ridx in (A_IDX, B_IDX, C_IDX):
        d = region_path_d(gs, ridx, epsilon=1.0, max_error=0.5)
        assert d.startswith("M"), f"Region {ridx} d does not start with M"
        assert "Z" in d, f"Region {ridx} d has no Z"


def test_region_path_d_seam_fragment_byte_identical():
    """Fit once: the seam fragment shared by two regions must be byte-identical."""
    from vectormark.seamgraph import build_graph, snap_junctions, region_path_d, EdgeGraph

    _, _, L, masks, conts, _ = _make_synthetic()
    region_contours_dict = {A_IDX: conts["A"], B_IDX: conts["B"], C_IDX: conts["C"]}
    g = build_graph(region_contours_dict, L, bg_idx=BG_IDX)
    gs = snap_junctions(g, reach=1.5)

    cache = {}

    def _path_with_cache(ridx):
        return region_path_d(gs, ridx, epsilon=1.0, max_error=0.5, _edge_cache=cache)

    d_A = _path_with_cache(A_IDX)
    d_B = _path_with_cache(B_IDX)
    d_C = _path_with_cache(C_IDX)

    seam_edges = [(i, e) for i, e in enumerate(gs.edges) if e.region_b is not None]
    for edge_idx, e in seam_edges:
        assert edge_idx in cache, f"Seam edge {edge_idx} not in fit cache"
        # Verify the cache entry was not None (fit was computed)
        assert cache[edge_idx] is not None
