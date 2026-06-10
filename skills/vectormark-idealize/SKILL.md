---
name: vectormark-idealize
description: >-
  Turn a raster logo image into a clean, editable, exactly-symmetric SVG using
  the vectormark CLI. Use when the user wants to vectorize, idealize, clean up,
  trace, or convert a logo/icon/emblem image (PNG/JPG) to SVG — including an
  image the agent just generated. Best on flat-color marks, app icons, and
  simple logos.
---

# Idealize a logo into SVG with vectormark

vectormark is a deterministic raster-logo → SVG idealizer. This skill takes a
raster image — one the user supplies, or one you just generated — and produces a
clean, editable, symmetric SVG by running the `vectormark` command-line tool.

The handoff is a **file path**, not image bytes: you save the raster to a file,
then point the CLI at it. (A path is short text you can pass between steps; raw
image bytes are not.)

## Step 1 — Decide how to run vectormark

vectormark is published on PyPI. Pick the invocation that fits the environment;
you will reuse it as `<VM>` in Step 3.

**Preferred — `uvx` (no install, ephemeral).** If `uv` is available
(`uv --version` succeeds), run vectormark with no persistent install — `uvx`
fetches and caches it on first use, later runs reuse the cache. **Prefer the
`scoring` extra**, which enables render-ΔE candidate scoring for the best output:

```bash
# <VM> = full, with scoring (preferred)
uvx --from "vectormark[scoring]" vectormark
# <VM> = minimal (no resvg-py; cascade-priority pick, warns once)
uvx vectormark
```

**Fallback — `pip install` (persistent).** If `uv` is not available but Python
≥ 3.12 and `pip` are, install once, then call `vectormark` directly:

```bash
pip install "vectormark[scoring]"   # or: pip install vectormark   (minimal)
# <VM> = vectormark
```

If the scoring extra can't be fetched (a locked-down sandbox with no `resvg-py`
wheel), drop `[scoring]` — vectormark still works, just with the cascade pick.

> Notes
> - Requires Python ≥ 3.12. vectormark's own wheel is pure-Python; its
>   dependencies and the `scoring` extra ship prebuilt manylinux/musllinux/
>   macOS/Windows wheels, so this works in any networked environment.
> - **No PyPI?** Install from the git source instead. Minimal:
>   `uvx --from "git+https://github.com/pradeepmouli/vectormark" vectormark`.
>   With scoring, use the PEP 508 form so the extra is carried:
>   `uvx --from "vectormark[scoring] @ git+https://github.com/pradeepmouli/vectormark" vectormark`
>   (same `"<spec> @ git+…"` form for `pip install`).

## Step 2 — Get the raster onto disk

You need a PNG/JPG file path. Two cases:

- **The user already gave you an image file** → use that path; skip to Step 3.
- **You are generating the logo** (image-gen tool / model) → generate it, then
  **save the bytes to a file**, e.g. `/tmp/vm-logo.png`. Some hosts already
  expose the generated image as a file in the working sandbox; if so, use that
  path. Otherwise write the bytes out explicitly before Step 3.

The CLI reads the file from disk, so the generator and `vectormark` must share a
filesystem — true when both run locally, or both run inside the same sandbox.

## Step 3 — Run vectormark

Use the `<VM>` invocation you chose in Step 1:

```bash
# preferred (uvx + scoring):
uvx --from "vectormark[scoring]" vectormark /tmp/vm-logo.png -o /tmp/vm-logo.svg --colors 16
# or, after a pip install:
vectormark /tmp/vm-logo.png -o /tmp/vm-logo.svg --colors 16
```

The SVG is written to `-o`; omit `-o` to print it to stdout (handy when you want
to read the SVG back to the user directly).

Useful flags (all optional):

| Flag | Default | Meaning |
|------|---------|---------|
| `-o, --output` | stdout | write the SVG to this path |
| `--colors` | 16 | maximum palette colors |
| `--epsilon` | 1.5 | primitive/polygon recognition tolerance, px |
| `--max-error` | 1.0 | Bézier fit tolerance, px |
| `--flatten` | off | emit plain paths instead of native primitives + `<use>` |
| `--no-symmetry` | off | disable symmetry detection |

Guidance: start with defaults. Raise `--colors` for richer marks; lower it to
force a flatter palette. Raise `--epsilon` if curves are coming out too busy.
Use `--flatten` only if the consumer can't handle `<use>`/native primitives.

## Step 4 — Deliver the result

- **The deliverable is the SVG itself** — report it and its output path, nothing
  more. vectormark emits one SVG string; do **not** fabricate optimization stats,
  file-size tables, color-swatch grids, "production ready" checklists, or a
  multi-panel contact sheet around it. Those are invented, not produced by the
  tool, and only add noise.
- **If you can't run vectormark, do not try to draw the SVG yourself.** Never
  hand-author, guess, or hallucinate an output. If the tool didn't actually run
  (not installed, no network, an error, or you only have the input image), say so
  and stop — a fabricated SVG is worse than none, and only vectormark's real
  output is meaningful here.
- To preview, open the file (`open /tmp/vm-logo.svg` on macOS,
  `xdg-open` on Linux) or read the SVG text back to the user — an `<svg>`
  document renders directly in any browser or markdown preview.

## Tips

- vectormark is **deterministic**: same input + flags → same SVG every time.
- It targets flat-color logos/icons/emblems. Photographic or heavily-gradient
  images are out of scope and will produce poor results.
- If you generated the image, keep the prompt simple and the mark flat — that is
  what idealizes cleanly.
