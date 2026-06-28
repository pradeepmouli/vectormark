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
