"""Seam-graph: shared-edge planar map for gap-free adjacent-region fitting.

Phase A: pure module — no pipeline integration.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Contour side-label classification
# ---------------------------------------------------------------------------

def _bilinear_sample(L: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly sample (H,W,K) array at sub-pixel point (x, y)."""
    H, W, K = L.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    dx = x - np.floor(x); dy = y - np.floor(y)
    return ((1 - dy) * (1 - dx) * L[y0c, x0c]
            + (1 - dy) * dx * L[y0c, x1c]
            + dy * (1 - dx) * L[y1c, x0c]
            + dy * dx * L[y1c, x1c])


def classify_contour(
    contour: np.ndarray,
    L: np.ndarray,
    region_idx: int,
    *,
    bg_idx: int,
) -> np.ndarray:
    """For each point of contour (N,2) (x,y), return the integer label across
    the boundary — the argmax of L at that sub-pixel point among labels ≠ region_idx.
    Value-ordered argmax ties (prefer lower index)."""
    K = L.shape[2]
    N = len(contour)
    result = np.empty(N, dtype=np.intp)
    bias = np.arange(K) * 1e-12          # value-ordered tie-break: prefer lower idx
    for i in range(N):
        x, y = float(contour[i, 0]), float(contour[i, 1])
        lv = _bilinear_sample(L, x, y) + bias
        lv[region_idx] = -np.inf
        result[i] = int(np.argmax(lv))
    return result


# ---------------------------------------------------------------------------
# Task 2: Planar edge graph
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Edge:
    pts: np.ndarray          # (N,2) points along this edge, forward direction
    region_a: int            # first (lower-index) region
    region_b: Optional[int]  # second region, or None for boundary vs background
    node0: int               # index into EdgeGraph.nodes for pts[0]
    node1: int               # index into EdgeGraph.nodes for pts[-1]


@dataclass
class EdgeGraph:
    nodes: list[tuple[float, float]] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def _round6(v: float) -> float:
    return round(float(v), 6)


def _pt_key(pt) -> tuple[float, float]:
    return (_round6(pt[0]), _round6(pt[1]))


def _coord_key(pts: np.ndarray) -> frozenset:
    """Canonical (direction-independent) key for a run of points.
    Uses frozenset so duplicate vertices (contour wrap artifacts) don't break dedup."""
    return frozenset((_round6(p[0]), _round6(p[1])) for p in pts)


def _split_runs(contour: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a closed contour into maximal runs of constant label.
    Returns list of (start_idx, end_idx, label) where indices are into contour.
    The contour is treated as a closed loop (last point connects back to first).
    """
    N = len(contour)
    if N == 0:
        return []

    # Find transition indices (where label changes)
    transitions = []
    for i in range(N):
        if labels[i] != labels[(i - 1) % N]:
            transitions.append(i)

    if not transitions:
        # Entire contour is one run
        return [(0, N - 1, int(labels[0]))]

    runs = []
    for k, t in enumerate(transitions):
        start = t
        end = (transitions[(k + 1) % len(transitions)] - 1) % N
        lbl = int(labels[start])
        runs.append((start, end, lbl))
    return runs


def _extract_run_pts(contour: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract contour points from start to end (inclusive), wrapping around."""
    N = len(contour)
    if start <= end:
        return contour[start:end + 1]
    # Wrap around
    return np.vstack([contour[start:], contour[:end + 1]])


def _node_index(
    nodes: list[tuple[float, float]],
    node_map: dict[tuple[float, float], int],
    pt,
) -> int:
    key = _pt_key(pt)
    if key not in node_map:
        node_map[key] = len(nodes)
        nodes.append(key)
    return node_map[key]


def build_graph(
    region_contours: dict[int, np.ndarray],
    L: np.ndarray,
    *,
    bg_idx: int,
) -> EdgeGraph:
    """Classify each region's contour, split into maximal constant-label runs,
    dedup seam runs by exact coordinate match, and build the planar edge graph."""

    # Collect all runs from all regions
    all_runs: list[tuple[int, int, np.ndarray]] = []
    for ridx, contour in sorted(region_contours.items()):
        labels = classify_contour(contour, L, ridx, bg_idx=bg_idx)
        runs = _split_runs(contour, labels)
        for (start, end, lbl) in runs:
            pts = _extract_run_pts(contour, start, end)
            if len(pts) < 2:
                continue
            all_runs.append((ridx, lbl, pts))

    # Build node list and edge list, deduplicating seam runs
    nodes: list[tuple[float, float]] = []
    node_map: dict[tuple[float, float], int] = {}
    edges: list[Edge] = []
    seam_key_to_edge_idx: dict[frozenset, int] = {}

    for ridx, other_lbl, pts in all_runs:
        is_seam = other_lbl != bg_idx
        ck = _coord_key(pts)

        if is_seam and ck in seam_key_to_edge_idx:
            # Seam already stored from the other region — skip duplicate
            continue

        n0 = _node_index(nodes, node_map, pts[0])
        n1 = _node_index(nodes, node_map, pts[-1])

        if is_seam:
            r_a = min(ridx, other_lbl)
            r_b = max(ridx, other_lbl)
            edge = Edge(pts=pts, region_a=r_a, region_b=r_b, node0=n0, node1=n1)
            idx = len(edges)
            edges.append(edge)
            seam_key_to_edge_idx[ck] = idx
        else:
            edge = Edge(pts=pts, region_a=ridx, region_b=None, node0=n0, node1=n1)
            edges.append(edge)

    return EdgeGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Task 3: Junction snapping
# ---------------------------------------------------------------------------

def snap_junctions(graph: EdgeGraph, *, reach: float = 1.5) -> EdgeGraph:
    """Cluster graph nodes within `reach` px where ≥3 edges are incident,
    replace each cluster with its centroid, and move incident edge endpoints.
    Uses a deterministic union-find sorted by (x, y) position."""

    nodes = list(graph.nodes)
    edges = list(graph.edges)
    n = len(nodes)

    # Count edge incidence per node
    incidence: list[int] = [0] * n
    for e in edges:
        incidence[e.node0] += 1
        incidence[e.node1] += 1

    # Deterministic union-find: process candidates sorted by position
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        # Value-ordered: lower index becomes root (deterministic)
        if ri < rj:
            parent[rj] = ri
        else:
            parent[ri] = rj

    # Collect candidate nodes (those with ≥3 incident edges or close to such nodes)
    # Sort by position for determinism
    sorted_indices = sorted(range(n), key=lambda i: (nodes[i][0], nodes[i][1]))

    for ii in range(len(sorted_indices)):
        for jj in range(ii + 1, len(sorted_indices)):
            i, j = sorted_indices[ii], sorted_indices[jj]
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if xj - xi > reach:
                break  # sorted by x — once x-gap exceeds reach, all later nodes are also beyond
            if dist <= reach:
                union(i, j)

    # Group nodes by root
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # Only snap clusters where ≥3 edges incident across the cluster
    new_nodes = list(nodes)
    node_remap: dict[int, int] = {}  # old node idx → new node idx

    # Build list of new (unique) nodes
    final_nodes: list[tuple[float, float]] = []
    old_to_new: dict[int, int] = {}

    # Process clusters
    for root, members in sorted(clusters.items()):
        cluster_incidence = sum(incidence[m] for m in members)
        if len(members) >= 2 and cluster_incidence >= 3:
            # Snap to centroid
            cx = sum(nodes[m][0] for m in members) / len(members)
            cy = sum(nodes[m][1] for m in members) / len(members)
            new_idx = len(final_nodes)
            final_nodes.append((_round6(cx), _round6(cy)))
            for m in members:
                old_to_new[m] = new_idx
        else:
            # Keep each node as-is
            for m in members:
                new_idx = len(final_nodes)
                final_nodes.append(nodes[m])
                old_to_new[m] = new_idx

    # Rebuild edges with remapped node indices, adjusting endpoint pts
    new_edges: list[Edge] = []
    for e in edges:
        n0 = old_to_new[e.node0]
        n1 = old_to_new[e.node1]
        pts = e.pts.copy()
        # Update first/last point to match snapped node position
        pts[0] = np.array(final_nodes[n0])
        pts[-1] = np.array(final_nodes[n1])
        new_edges.append(Edge(
            pts=pts,
            region_a=e.region_a,
            region_b=e.region_b,
            node0=n0,
            node1=n1,
        ))

    return EdgeGraph(nodes=final_nodes, edges=new_edges)


# ---------------------------------------------------------------------------
# Task 5: Region reassembly → path d
# ---------------------------------------------------------------------------

from vectormark.fit import fit_edge as _fit_edge_fn


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _edge_svg_fragment_forward(
    edge: Edge, beziers: list, *, start_pt: tuple
) -> str:
    """SVG fragment for this edge traversed forward (pts[0]→pts[-1]).
    Returns commands after the M (caller already emitted M for pts[0])."""
    _ = start_pt  # consumed by caller's M
    return _beziers_to_fragment(beziers, edge.pts[-1])


def _edge_svg_fragment_reversed(
    edge: Edge, beziers: list, *, start_pt: tuple
) -> str:
    """SVG fragment for this edge traversed reversed (pts[-1]→pts[0])."""
    _ = start_pt
    return _beziers_to_fragment_reversed(beziers, edge.pts[0])


def _beziers_to_fragment(beziers: list, end_pt: np.ndarray) -> str:
    """Emit bezier list as SVG fragment ending at end_pt. Straight → L, curve → C."""
    if not beziers:
        # Straight line (fit_edge emitted an L — re-emit to end_pt)
        return f"L{_fmt(end_pt[0])} {_fmt(end_pt[1])} "
    d = ""
    for b in beziers:
        d += (f"C{_fmt(b[1][0])} {_fmt(b[1][1])} "
              f"{_fmt(b[2][0])} {_fmt(b[2][1])} "
              f"{_fmt(b[3][0])} {_fmt(b[3][1])} ")
    return d


def _beziers_to_fragment_reversed(beziers: list, end_pt: np.ndarray) -> str:
    """Emit reversed bezier list (each cubic reversed + list reversed)."""
    if not beziers:
        return f"L{_fmt(end_pt[0])} {_fmt(end_pt[1])} "
    d = ""
    for b in reversed(beziers):
        # Reverse cubic: swap P0↔P3, P1↔P2
        rev = np.array([b[3], b[2], b[1], b[0]])
        d += (f"C{_fmt(rev[1][0])} {_fmt(rev[1][1])} "
              f"{_fmt(rev[2][0])} {_fmt(rev[2][1])} "
              f"{_fmt(rev[3][0])} {_fmt(rev[3][1])} ")
    return d


def _fit_edge_beziers(edge: Edge, *, epsilon: float, max_error: float) -> list:
    """Fit edge.pts as an open arc; return list of (4,2) bezier arrays.
    Straight edges return []."""
    from vectormark.fit import _segment_is_straight
    pts = edge.pts
    if _segment_is_straight(pts, epsilon):
        return []
    frag = _fit_edge_fn(pts, epsilon=epsilon, max_error=max_error)
    # Parse C commands from the fragment string
    beziers = []
    # We need to reconstruct bezier arrays from the fragment.
    # Since fit_edge returns the complete fragment, re-run fit_cubic_beziers
    from vectormark.contour import rdp
    from vectormark._fitcurve import fit_cubic_beziers
    from vectormark.fit import PATH_DENOISE_EPS
    run = rdp(pts, min(PATH_DENOISE_EPS, max_error)) if len(pts) > 4 else pts
    beziers = list(fit_cubic_beziers(run, max_error))
    return beziers


def _order_edges_for_region(
    graph: EdgeGraph, region_idx: int
) -> list[tuple[int, bool]]:
    """Return list of (edge_index, forward) pairs forming a closed walk for region_idx.
    forward=True means edge traversed pts[0]→pts[-1]; False means pts[-1]→pts[0].
    Raises if a closed walk cannot be formed."""

    # Collect incident edges and their traversal direction for this region
    incident: list[tuple[int, bool]] = []
    for i, e in enumerate(graph.edges):
        if e.region_a == region_idx or e.region_b == region_idx:
            # Determine traversal direction:
            # For a seam (region_a < region_b), forward means the contour of region_a
            # traced the edge in its natural direction. region_b uses it reversed.
            if e.region_b is None:
                # Boundary edge: always forward for its owner region
                forward = True
            elif e.region_a == region_idx:
                forward = True
            else:
                forward = False
            incident.append((i, forward))

    if not incident:
        return []

    # Build node-to-(edge_idx, forward) adjacency for this region
    # Each edge contributes two endpoints based on traversal direction
    def edge_endpoints(edge_idx: int, forward: bool) -> tuple[int, int]:
        e = graph.edges[edge_idx]
        if forward:
            return e.node0, e.node1
        else:
            return e.node1, e.node0

    # Topological sort: build closed walk
    # Map: tail_node → list of (edge_idx, forward)
    from collections import defaultdict
    tail_to_edge: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for ei, fwd in incident:
        tail, _ = edge_endpoints(ei, fwd)
        tail_to_edge[tail].append((ei, fwd))

    # Start from smallest edge index for determinism
    start_ei, start_fwd = min(incident, key=lambda x: x[0])
    start_tail, start_head = edge_endpoints(start_ei, start_fwd)

    walk: list[tuple[int, bool]] = [(start_ei, start_fwd)]
    used = {start_ei}
    current_node = start_head

    while current_node != start_tail:
        candidates = [
            (ei, fwd) for (ei, fwd) in tail_to_edge[current_node]
            if ei not in used
        ]
        if not candidates:
            break
        # Deterministic: pick smallest edge index
        next_ei, next_fwd = min(candidates, key=lambda x: x[0])
        walk.append((next_ei, next_fwd))
        used.add(next_ei)
        _, current_node = edge_endpoints(next_ei, next_fwd)

    return walk


def region_path_d(
    graph: EdgeGraph,
    region_idx: int,
    *,
    epsilon: float,
    max_error: float,
    _edge_cache: dict | None = None,
) -> str:
    """Assemble a closed SVG path d for region_idx from the shared edge graph.
    Each edge is fitted once (cached by edge index). Shared seams: both regions
    use the same bezier list (reversed as needed) → gap-free invariant.
    """
    if _edge_cache is None:
        _edge_cache = {}

    walk = _order_edges_for_region(graph, region_idx)
    if not walk:
        return ""

    d = ""
    first = True
    for edge_idx, forward in walk:
        edge = graph.edges[edge_idx]

        # Fit once, cache by edge index
        if edge_idx not in _edge_cache:
            _edge_cache[edge_idx] = _fit_edge_beziers(
                edge, epsilon=epsilon, max_error=max_error
            )
        beziers = _edge_cache[edge_idx]

        if forward:
            start_pt = edge.pts[0]
            end_pt = edge.pts[-1]
        else:
            start_pt = edge.pts[-1]
            end_pt = edge.pts[0]

        if first:
            d += f"M{_fmt(start_pt[0])} {_fmt(start_pt[1])} "
            first = False

        if forward:
            d += _beziers_to_fragment(beziers, end_pt)
        else:
            d += _beziers_to_fragment_reversed(beziers, end_pt)

    d += "Z"
    return d
