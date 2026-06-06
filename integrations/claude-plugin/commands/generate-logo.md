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
2. **Generate** a raster logo with whatever image-generation tool is available
   (image model/API, Canva MCP, etc.), saving it to a local PNG. Prompt for a
   flat, few-color, centered mark on a solid background, square, ≥ 1024×1024.
3. **Idealize** it through the vectormark CLI:
   `vectormark ./logo-raster.png -o ./logo.svg`
   (fall back to `uv run vectormark ...` if not on PATH).
4. Verify `./logo.svg` exists and is non-empty, report the path, and offer one
   refinement loop (palette / `--no-symmetry` / `--flatten`).

If no image-generation tool is available, ask the user to supply a raster
instead of fabricating SVG by hand. If vectormark isn't installed, do not skip
idealization — tell the user how to install it (see the skill's reference.md).
