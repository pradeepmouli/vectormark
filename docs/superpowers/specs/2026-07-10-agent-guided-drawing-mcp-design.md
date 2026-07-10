# Agent-Guided Drawing MCP Design

**Date:** 2026-07-10  
**Area:** `src/vectormark/mcp_server.py` plus new trace, plan, and session-state modules.

## Goal

Expose an agent-guided drawing workflow through the MCP server. VectorMark first
traces a raster into stable, labeled raw paths. An agent then sends a semantic
refinement plan. VectorMark executes the plan against the retained drawing state
and returns a clean SVG.

```text
trace_drawing(image, options)
  -> drawing_id + version v0 + raw regions + labeled region map

refine_drawing(plan)
  -> drawing_id + derived version + SVG + refinement report + preview
```

## Product boundary

This MVP is stateful and MCP-server-local. `trace_drawing` retains the
original resolved image bytes, processed RGB array, image metadata, segmented
regions, masks, contour samples, trace paths, and command IDs in memory. It returns an
opaque `drawing_id`; `refine_drawing` accepts no image and reads the ID from
the plan.

Each drawing belongs to the originating FastMCP `Context.session`. A drawing ID
is cryptographically random, valid only within that session, and expires after
30 minutes of inactivity. A successful refinement refreshes its idle expiry.
Unknown, expired, or cross-session IDs return the structured error code
`DRAWING_NOT_FOUND`. The MVP never writes drawing state to disk and does not
provide a portable trace import/export format.

## Trace contract

`trace_drawing` uses the existing secure image resolver and preprocessor,
then performs only deterministic palette segmentation, contour extraction, and
path fitting. It does **not** run primitive recognition, symmetry detection,
surface merging, fill inference, clone detection, or optimizer passes.

The public path is named `trace_path`, not `raw_path`. The stored contour samples
are the raw pixel or coverage evidence; `trace_path` is a pre-simplified line and
Bézier representation of that evidence. It is deliberately free of semantic
interpretation while remaining compact enough for stable command IDs and agent
reasoning.

### Trace options

```json
{
  "max_colors": 16,
  "min_region_size": 16,
  "trace_level": "pixel",
  "simplify_tolerance": 1.5,
  "curve_tolerance": 1.0,
  "curve_type": "quadratic"
}
```

- `max_colors` is the palette-quantization ceiling.
- `min_region_size` is the minimum retained connected-component pixel count.
  No relative-to-largest-region cutoff is applied in this workflow.
- `trace_level` is `pixel` for binary quantized-mask contours or `subpixel` for
  anti-alias-aware coverage contours. The result reports each region's effective
  level because the coverage safety guard may fall back to `pixel`.
- `simplify_tolerance` controls contour simplification and straight-run detection.
- `curve_tolerance` bounds fitted Bézier error.
- `curve_type` is `quadratic` by default or `cubic` for intentionally complex
  contours. Quadratics are the default because they resist raster stair-stepping.

Input crop, resize, and alpha handling remain in the existing preprocess options.
There are no trace options for symmetry, primitive recognition, fill inference,
surface merging, flattening, or shadow removal.

The returned structured result contains:

```json
{
  "drawing_id": "drw_<opaque>",
  "version": "v0",
  "canvas": { "width": 1024, "height": 1024, "view_box": [0, 0, 1024, 1024] },
  "regions": [
    {
      "id": "r1",
      "color": "#0074F0",
      "area": 42120,
      "bbox": [172, 240, 540, 670],
      "trace_path": {
        "d": "M…Z",
        "fill_rule": "nonzero",
        "commands": [
          { "id": "r1.p0.c0", "command": "M", "values": [172, 240] },
          { "id": "r1.p0.c1", "command": "Q", "values": [310, 242, 390, 260] }
        ]
      }
    }
  ],
  "region_map_svg": "<svg …/>"
}
```

Region IDs are deterministic for one trace: sort retained regions by
descending area, then bounding-box top, bounding-box left, color, and original
segment label. Command IDs include a subpath index so holes and disconnected
contours remain unambiguous.

The region-map SVG renders every raw region with a translucent fill and its
region ID centered in the region bounding box. It is a diagnostic artifact for
the agent, not final artwork.

## Refinement-plan contract

`refine_drawing` accepts a single typed plan:

```json
{
  "version": "vectormark.plan.v1",
  "drawing_id": "drw_<opaque>",
  "base_version": "v0",
  "label": "optional user-facing description",
  "ops": []
}
```

The server validates the plan before mutation: the drawing exists for the
current session; the base version exists; every referenced region, group,
command, and fill is known;
operation IDs are unique; and path source commands form a directed contiguous
run within one raw subpath. Validation errors are returned as `INVALID_PLAN`
with a JSON pointer and actionable message.

### Version tree

Every trace begins at immutable version `v0`. A successful plan creates a new
immutable child of its base version; it never overwrites its parent. Version IDs
are branch paths: the first, second, and third refinements of `v0` are `v0.0`,
`v0.1`, and `v0.2`; the first child of `v0.1` is `v0.1.0`. The server assigns the
next child segment atomically, so agents provide only `base_version` and cannot
collide when submitting alternatives concurrently.

Each version stores its parent version, the accepted plan, derived render
targets, SVG, report, preview availability, and the optional plan `label`.
Agents can branch from any retained version and present sibling alternatives to
users without retracing the original raster.

### MVP region operations

- `group`: create a semantic group from regions without altering their
  boundaries or paint order.
- `set_geometry`: assign a target to `circle`, `ellipse`, `rect`, `polygon`, or
  `path`.
- `set_fill`: assign `flat`, `linear_gradient`, `radial_gradient`, or `raster`.
- `set_z_order`: establish final paint order for all emitted targets.

`group` is intentionally not boolean union. A future explicit boolean operation
will create a new derived boundary and command-ID namespace; silently treating a
group as a union would make the agent's raw command references invalid.

### MVP path operations

For `set_geometry` with `type: "path"`, the geometry owns an ordered `ops`
array:

- `group`: trace-path commands -> named logical segment.
- `fit`: logical segment -> `line`, `quadratic`, `cubic`, or `keep`.
- `break`: request a G0 discontinuity after a named segment.
- `close`: close the current subpath.

Adjacent fitted path segments are tangent-continuous by default; only `break`
requests a sharp join. A `fit` operation creates final SVG commands; agents do
not author final path coordinates.

## Execution model

The trace module produces an internal `DrawingState` and a public,
JSON-safe summary. The plan executor creates fresh render targets from this
immutable state, fits requested geometry to its raw contour evidence, maps
declared fills to the existing fill classes, and serializes through the existing
SVG emitter. It does not mutate the stored trace. Each accepted plan creates an
immutable version, so the agent may issue multiple branches for one drawing.

The refinement result includes its new version ID, final SVG, preview
availability, target-level geometry/fill summaries, and residual measurements
against the stored raster masks. It also returns warnings for lossy fits without
silently falling back to automatic idealization.

### Trace-engine boundary and native follow-up

The tracing implementation is VectorMark's own pipeline: palette segmentation,
`find_contours`, and custom line/Bézier fitting. It is not Potrace or vtracer.
The MCP MVP wraps it behind a `TraceEngine.trace(rgb, options) -> TraceResult`
boundary. The initial implementation uses the existing Python/scikit-image
pipeline; retained `DrawingState` ensures every branch refines a trace once
rather than retracing it.

Trace-engine substitution is a Phase 2 performance project, after profiling
representative MCP drawings. A Rust implementation or an adapter for a
third-party tracer may replace segmentation, contour extraction, and curve
fitting behind the same `TraceEngine` interface. Any replacement must preserve
the public trace result, command-ID rules, and trace-options semantics, so it
does not alter the MCP tools, drawing store, version tree, or plan schema.

## Non-goals

- CLI commands or a portable trace file.
- Automatic primitive/candidate suggestions.
- Auto symmetry, clone inference, path congruency, constraints, ribbons, arcs,
  split regions, or boolean unions.
- Persistence across an MCP server restart or session end.
- Altering `idealize_logo` behavior.

## Testing and acceptance

Unit tests cover deterministic IDs, command parsing, session isolation, sliding
TTL, expiration, plan validation, primitive fitting, path operation execution,
fill emission, z-order, and no mutation of stored state. MCP integration tests
call both tools over stdio using a data URI.

Given a flat folded-ribbon-style raster, acceptance requires a labeled region
map, two sibling plans from `v0` with distinct labels, a child plan from one
sibling, native circles for equal dots, a flat or gradient fill, and no automatic
symmetry or candidate inference during tracing.
