"""Corner provenance and conservative corner-local normalization."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .._fitcurve import fit_quadratic_once
from ..fit import Shape, _fmt, _quadratic_max_residual
from .vector_region import _parse_subpaths, _sample_subpath

_POINT_CORNER_TURN_DEGREES = 45.0
_CURVED_CORNER_TURN_DEGREES = 35.0
_CURVE_COMMANDS = {"Q", "C"}


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    return None if length <= 1e-9 else vector / length


def _turn_degrees(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _unit(left), _unit(right)
    if a is None or b is None:
        return 0.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))


def _segments(tokens: list[tuple[str, list[float]]]) -> list[dict[str, object]]:
    cursor: np.ndarray | None = None
    subpath_start: np.ndarray | None = None
    segments: list[dict[str, object]] = []
    for command_index, (command, values) in enumerate(tokens):
        if command == "M":
            cursor = np.asarray(values[:2], dtype=float)
            subpath_start = cursor.copy()
            continue
        if cursor is None:
            continue
        start = cursor.copy()
        if command == "Z":
            end = subpath_start.copy() if subpath_start is not None else start.copy()
            start_tangent = end - start
            end_tangent = start_tangent
        elif command == "L":
            end = np.asarray(values[:2], dtype=float)
            start_tangent = end - start
            end_tangent = start_tangent
        elif command == "Q":
            control = np.asarray(values[:2], dtype=float)
            end = np.asarray(values[2:4], dtype=float)
            start_tangent = control - start
            end_tangent = end - control
        elif command == "C":
            first = np.asarray(values[:2], dtype=float)
            second = np.asarray(values[2:4], dtype=float)
            end = np.asarray(values[4:6], dtype=float)
            start_tangent = first - start
            end_tangent = end - second
        else:
            continue
        segments.append(
            {
                "command_index": command_index,
                "command": command,
                "start": start,
                "end": end,
                "start_tangent": start_tangent,
                "end_tangent": end_tangent,
            }
        )
        cursor = end
    return segments


def path_corner_diagnostics(d: str) -> dict[str, object]:
    """Classify command-end anchors that are part of point or curved corners.

    The result is intentionally command-indexed: an SVG command owns the
    anchor at its endpoint, so there is no parallel anchor-address namespace.
    Curved-corner members share a ``corner_id`` and include both boundary
    anchors plus every curve command in the detected run.
    """
    anchors: dict[tuple[int, int], dict[str, object]] = {}
    spans: list[dict[str, object]] = []
    corner_number = 0
    total_commands = 0

    def mark(subpath: int, command_index: int, corner_id: str) -> None:
        anchors[(subpath, command_index)] = {
            "subpath": subpath,
            "command_index": command_index,
            "anchor_kind": "corner",
            "corner_id": corner_id,
        }

    for subpath_index, tokens in enumerate(_parse_subpaths(d)):
        total_commands += sum(command not in {"M", "Z"} for command, _values in tokens)
        segments = _segments(tokens)
        if not segments:
            continue

        # A sharp L/L junction is already the canonical point-corner form.
        for left, right in zip(segments, segments[1:], strict=False):
            if left["command"] != "L" or right["command"] not in {"L", "Z"}:
                continue
            if _turn_degrees(left["end_tangent"], right["start_tangent"]) < _POINT_CORNER_TURN_DEGREES:
                continue
            owner = int(left["command_index"])
            corner_id = f"p{subpath_index}.corner{corner_number}"
            corner_number += 1
            mark(subpath_index, owner, corner_id)
            spans.append({
                "corner_id": corner_id,
                "kind": "point",
                "subpath": subpath_index,
                "commands": [owner],
            })

        # The closing Z owns the initial M anchor.  Its outgoing join has no
        # independent SVG segment record, so annotate M when the incoming
        # closing edge and outgoing first edge form a point.
        drawable = [segment for segment in segments if segment["command"] != "Z"]
        if (
            tokens
            and tokens[-1][0] == "Z"
            and len(drawable) >= 2
            and drawable[0]["command"] == "L"
            and segments[-1]["command"] == "Z"
            and _turn_degrees(drawable[-1]["end_tangent"], drawable[0]["start_tangent"])
            >= _POINT_CORNER_TURN_DEGREES
        ):
            corner_id = f"p{subpath_index}.corner{corner_number}"
            corner_number += 1
            mark(subpath_index, 0, corner_id)
            spans.append({
                "corner_id": corner_id,
                "kind": "point",
                "subpath": subpath_index,
                "commands": [0],
            })

        # A curve run between straight sides is a rounded-corner candidate.
        index = 0
        while index < len(segments):
            if segments[index]["command"] not in _CURVE_COMMANDS:
                index += 1
                continue
            end = index
            while end + 1 < len(segments) and segments[end + 1]["command"] in _CURVE_COMMANDS:
                end += 1
            previous = segments[index - 1] if index else None
            following = segments[end + 1] if end + 1 < len(segments) else None
            if previous is None or following is None or previous["command"] != "L" or following["command"] != "L":
                index = end + 1
                continue
            run = segments[index:end + 1]
            turn = sum(
                _turn_degrees(segment["start_tangent"], segment["end_tangent"])
                for segment in run
            )
            if turn < _CURVED_CORNER_TURN_DEGREES:
                index = end + 1
                continue
            corner_id = f"p{subpath_index}.corner{corner_number}"
            corner_number += 1
            member_commands = [int(previous["command_index"]), *(int(segment["command_index"]) for segment in run)]
            for command_index in member_commands:
                mark(subpath_index, command_index, corner_id)
            spans.append({
                "corner_id": corner_id,
                "kind": "curve",
                "subpath": subpath_index,
                "commands": member_commands,
                "curve_commands": [int(segment["command_index"]) for segment in run],
                "turn_degrees": round(turn, 3),
            })
            index = end + 1

    ordered_anchors = [anchors[key] for key in sorted(anchors)]
    corner_commands = {
        (int(anchor["subpath"]), int(anchor["command_index"]))
        for anchor in ordered_anchors
        if int(anchor["command_index"]) != 0
    }
    return {
        "commands": {
            "total": total_commands,
            "corner": len(corner_commands),
            "free": total_commands - len(corner_commands),
        },
        "anchor_counts": {
            "total": total_commands + sum(1 for tokens in _parse_subpaths(d) if tokens),
            "corner": len(ordered_anchors),
        },
        "anchors": ordered_anchors,
        "spans": spans,
    }


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    return " ".join(
        command if not values else f"{command}{' '.join(_fmt(value) for value in values)}"
        for command, values in tokens
    )


def normalize_corners_path_d(d: str, *, max_error: float) -> tuple[str, dict[str, object]]:
    """Collapse recognised rounded-corner runs to one Q when the fit is faithful."""
    diagnostics = path_corner_diagnostics(d)
    tokens_by_subpath = _parse_subpaths(d)
    normalized_spans: list[str] = []
    curve_spans = sorted(
        (span for span in diagnostics["spans"] if span["kind"] == "curve"),
        key=lambda span: (int(span["subpath"]), max(int(command) for command in span["curve_commands"])),
        reverse=True,
    )
    for span in curve_spans:
        subpath_index = int(span["subpath"])
        curve_commands = [int(command) for command in span["curve_commands"]]
        tokens = tokens_by_subpath[subpath_index]
        start, end = min(curve_commands), max(curve_commands)
        if not all(tokens[index][0] in _CURVE_COMMANDS for index in range(start, end + 1)):
            continue
        first_start = _segments(tokens)
        segment_by_command = {int(segment["command_index"]): segment for segment in first_start}
        source = [
            ("M", [float(value) for value in segment_by_command[start]["start"]]),
            *tokens[start:end + 1],
        ]
        samples = np.asarray(_sample_subpath(source, 16), dtype=float)
        if len(samples) < 3:
            continue
        quadratic, _squared_error, _split = fit_quadratic_once(samples, max_error)
        if _quadratic_max_residual(samples, quadratic) > max_error:
            continue
        end_point = quadratic[2]
        tokens[start:end + 1] = [
            ("Q", [float(quadratic[1][0]), float(quadratic[1][1]), float(end_point[0]), float(end_point[1])])
        ]
        normalized_spans.append(str(span["corner_id"]))

    normalized_d = " ".join(_subpath_d(tokens) for tokens in tokens_by_subpath)
    result = path_corner_diagnostics(normalized_d)
    result["normalized"] = normalized_spans
    return normalized_d, result


def anchor_annotations(diagnostics: Mapping[str, object], subpath: int) -> dict[int, dict[str, object]]:
    """Return trace-command annotations for one subpath."""
    anchors = diagnostics.get("anchors", ())
    if not isinstance(anchors, list):
        return {}
    return {
        int(anchor["command_index"]): dict(anchor)
        for anchor in anchors
        if isinstance(anchor, Mapping) and int(anchor.get("subpath", -1)) == subpath
    }
