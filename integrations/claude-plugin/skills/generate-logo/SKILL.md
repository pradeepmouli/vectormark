---
name: generate-logo
description: Use when the user wants to generate, create, design, or make a logo, brand mark, app icon, or emblem and wants a clean vector/SVG result. Produces a raster logo with whatever image-generation tooling is available, then idealizes it into a clean, symmetric, editable SVG via the vectormark CLI. Triggers on "generate a logo", "make me a logo", "design a brand mark", "create an app icon", "logo as SVG".
---

# Generate Logo → Idealize to SVG

Turn a logo idea into a clean, structured, editable SVG. This is a two-stage
pipeline:

1. **Generate** a raster (PNG) logo using a raster image-generation tool or
   model you have available (an image-generation model/API, a diffusion
   endpoint, etc. — tool-agnostic).
2. **Idealize** that raster into a clean SVG by piping it through the
   `vectormark` CLI. vectormark recognizes structure (palette, symmetry, ideal
   primitives) instead of emitting anonymous `<path>` soup, so the output is a
   small, human-editable, exactly-symmetric vector.

**Always route the raster through vectormark.** Do not hand the user a raw PNG
or a raw tracer dump as "the logo" — the deliverable is the idealized SVG.

## When to use this skill

Use it when the user asks to create/design/generate a **logo, brand mark, app
icon, emblem, or monogram** AND wants (or would benefit from) a vector/SVG
result. Also use it when the user already has a raster logo and wants it turned
into a clean SVG (skip straight to Stage 2).

Do **not** use it for: full illustrations, photos, complex scenes, or UI
mockups — vectormark targets flat-color segmented marks, not arbitrary images.

## Workflow

### Stage 0 — Brief (one short round)
Confirm the essentials in a single message if not already given: subject/concept,
style (geometric, wordmark, mascot, abstract), color palette, and whether
bilateral symmetry is desired. Keep it tight — one clarifying round, then build.

### Stage 1 — Generate the raster
Use the best raster image-generation tool or model you have. Prompt for
logo-friendly output:

- **Flat colors, few colors** (vectormark is built for flat-color segmented
  marks; gradients/photoreal degrade the idealization).
- **Solid background** (a plain white or transparent background; vectormark
  drops the background plate).
- **Centered, high contrast, no text effects/bevels/shadows** unless requested.
- **Square canvas, generous resolution** (≥ 1024×1024 is ideal).

Save the result to a **unique** local path — derive a slug from the brief
(e.g. `./fox-logo.png`) or add a timestamp — so re-runs don't overwrite an
earlier raster; don't clobber an existing file without asking. If your
generation tool returns a URL or an asset id, download/export it to a local PNG
first — vectormark needs a local raster file path.

If no image-generation tool is available, say so and ask the user to provide a
raster, or paste/attach one. Do not fabricate an SVG by hand.

### Stage 2 — Idealize with vectormark
Run the CLI to produce the SVG:

```bash
vectormark ./fox-logo.png -o ./fox-logo.svg
```

If `vectormark` is not on PATH, use the repo runner:

```bash
uv run vectormark ./fox-logo.png -o ./fox-logo.svg
```

Common flags (see `reference.md` for the full surface):
- `--colors N` — cap the palette (default 16). Lower it (e.g. `--colors 4`) for
  cleaner marks with few intended colors.
- `--no-symmetry` — disable symmetry detection if the mark is intentionally
  asymmetric.
- `--flatten` — emit plain `<path>`s instead of native primitives + `<use>`
  (use when the consumer can't handle `<use>`/`<ellipse>`/`<rect>`).
- `--epsilon PX` / `--max-error PX` — fit tolerances; raise to simplify,
  lower for fidelity.

### Stage 3 — Verify and deliver
- Confirm the output SVG exists and is non-empty.
- Sanity-check it opens (it's an SVG; you can inspect the markup — expect
  named shapes / a small file, not thousands of path points).
- Report the saved path to the user and offer one refinement loop (e.g.
  "want fewer colors / a different palette / asymmetric?"), re-running the
  appropriate stage.

## If vectormark isn't installed

See `reference.md` → "Installing vectormark". In short: it's a Python package
(`vectormark`), installable with `uv` or `pip` from the repo. If you cannot
install it, do NOT silently skip idealization — tell the user the SVG step is
blocked and how to install it.

## Details

For the full CLI reference, raster generation-tool guidance, troubleshooting,
and the exact handoff contract, read
[`reference.md`](reference.md).
