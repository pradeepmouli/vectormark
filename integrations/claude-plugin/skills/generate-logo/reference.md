# generate-logo — Reference

Loaded on demand by the `generate-logo` skill. Covers the full vectormark CLI
surface, generation-tool guidance, the handoff contract, and troubleshooting.

## The handoff contract

vectormark is a deterministic raster→SVG idealizer. The skill's job is to
**get a good local raster onto disk**, then call vectormark. Contract:

| Stage | Produces | Consumed by |
|-------|----------|-------------|
| Generate | a local PNG/JPG file (e.g. `./logo-raster.png`) | vectormark CLI |
| Idealize | a clean `.svg` (e.g. `./logo.svg`) | the user |

vectormark requires a **local file path** as input. URLs, asset ids, base64,
and in-memory images must be written to a local raster first.

## vectormark CLI surface

```
vectormark INPUT [-o OUTPUT] [--epsilon PX] [--max-error PX]
           [--colors N] [--flatten] [--no-symmetry]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `INPUT` | (required) | Input raster (PNG/JPG), a local path. |
| `-o`, `--output` | stdout | Output `.svg` path. Always pass `-o` so you get a file. |
| `--colors N` | 16 | Max palette colors. Lower for cleaner few-color marks. |
| `--epsilon PX` | 1.5 | Polyline fit tolerance (px). Higher = simpler. |
| `--max-error PX` | 1.0 | Bézier fit tolerance (px). Higher = simpler. |
| `--flatten` | off | Emit plain `<path>`s instead of primitives + `<use>`. |
| `--no-symmetry` | off | Disable symmetry detection (for asymmetric marks). |

Notes:
- Without `-o`, the SVG goes to **stdout** — for a saved file, always pass `-o`.
- vectormark targets **flat-color segmented marks**. Gradients/photoreal inputs
  produce poor idealizations — generate a flat, few-color mark instead.
- Output uses native `<ellipse>`/`<rect>`/regular polygons + geometry-level
  mirror via `<use>` when symmetric; `--flatten` collapses these to paths.

### Invocation variants

```bash
# Preferred: vectormark installed on PATH
vectormark ./logo-raster.png -o ./logo.svg

# Repo / dev checkout (uses the project's uv environment)
uv run vectormark ./logo-raster.png -o ./logo.svg

# Module form (if the console script isn't linked)
python -m vectormark.cli ./logo-raster.png -o ./logo.svg
```

## Installing vectormark

If `vectormark` is not found:

```bash
# From a checkout of the repo (recommended; pulls dev extras)
uv sync --extra dev          # then use: uv run vectormark ...

# Or install from the repo checkout into the current environment (editable)
uv pip install -e .          # or: pip install -e .
```

If installation isn't possible in the environment, **stop and tell the user**
the idealization step is blocked, point them at this section, and offer to
deliver the raster as an interim artifact (clearly labeled as not-yet-idealized).

## Choosing / using a generation tool (tool-agnostic)

Use whatever is available, in roughly this order of convenience:

1. **An image-generation MCP or model/API** you can call directly (e.g. an
   image model exposed as a tool). Prompt for a flat, few-color, centered logo
   on a solid background; save the returned image to a local PNG.
2. **The Canva MCP** (if connected): use its design-generation / export tools to
   produce a logo design, then **export it as PNG** and download to a local
   path. Canva tools typically return a design id and an export URL — fetch the
   URL to a local file before calling vectormark. Prefer a single, flat,
   logo-style page; avoid multi-element layouts.
3. **A local diffusion endpoint / CLI** if the user has one configured.
4. **User-supplied raster**: if the user already has/attaches an image, skip
   generation and go straight to vectormark.

Prompt guidance for any generator (maximizes idealization quality):
- "flat vector logo", "solid colors", "minimal", "centered", "plain white
  background", "no gradients, no shadows, no 3D, no text" (unless a wordmark is
  requested).
- Square aspect, ≥ 1024×1024.
- Keep the intended color count small and say it ("two-color", "monochrome").

If **no** generation tool exists, do not hand-write SVG markup as a substitute —
ask the user to provide a raster instead.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `vectormark: command not found` | not installed / not on PATH | use `uv run vectormark` or install (see above). |
| SVG is `<path>` soup / huge | gradient/photoreal input, or too many colors | regenerate flatter; lower `--colors`; raise `--epsilon`. |
| Symmetry looks forced/wrong | mark is intentionally asymmetric | re-run with `--no-symmetry`. |
| Colors merged/lost | palette cap too low | raise `--colors`. |
| Consumer can't render `<use>`/primitives | needs plain paths | re-run with `--flatten`. |
| Empty / error output | bad input path or unsupported format | confirm the local PNG/JPG path exists. |

## Quality checklist before delivering

- [ ] A local raster was generated/obtained and saved to disk.
- [ ] vectormark ran with `-o`, producing a non-empty `.svg`.
- [ ] The SVG is structured (named shapes / small file), not thousands of points.
- [ ] The saved path was reported to the user.
- [ ] One refinement loop offered (palette / symmetry / flatten).
