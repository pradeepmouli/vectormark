---
description: Generate a logo and idealize it into a clean SVG via vectormark
argument-hint: [logo concept + style/colors]
allowed-tools: [Read, Write, Bash, Glob, Grep]
---

# /generate-logo

The user invoked this command with: $ARGUMENTS

Use the **generate-logo** skill to run the full pipeline:

1. Treat `$ARGUMENTS` as the logo brief (concept, style, colors, symmetry). If
   it's empty or ambiguous, ask one short clarifying round, then proceed.
2. **Generate** a raster logo with a raster image-generation tool or model
   (an image-generation model/API, a diffusion endpoint, etc.), saving it to a
   unique local PNG — e.g. a slug from the brief, `./<slug>-logo.png` — rather
   than a fixed name, so repeated runs don't overwrite earlier output. Prompt
   for a flat, few-color, centered mark on a solid background, square, ≥ 1024×1024.
3. **Idealize** it through the vectormark CLI:
   `vectormark ./<slug>-logo.png -o ./<slug>-logo.svg`
   (fall back to `uv run vectormark ...` if not on PATH).
4. Verify the output SVG exists and is non-empty, report the path, and offer one
   refinement loop (palette / `--no-symmetry` / `--flatten`).

If no image-generation tool is available, ask the user to supply a raster
instead of fabricating SVG by hand. If vectormark isn't installed, do not skip
idealization — tell the user how to install it (see the skill's reference.md).
