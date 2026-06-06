# vectormark

> Deterministic logo idealizer — turn a *rendered* (raster) logo into a clean,
> editable, exactly-symmetric SVG.

Where conventional tracers (`vtracer`, `potrace`, `autotrace`) chase pixel edges
and emit anonymous `<path>` soup, **vectormark recognizes structure**: it
collapses anti-aliasing to a true palette, detects symmetry, and fits *ideal
primitives* (`<ellipse>`, `<rect>`, regular polygons) to regions — falling back
to fitted Bézier paths only where no primitive matches. The output SVG doubles as
a human-editable parametric model.

No model/LLM in the loop. Every stage is a known, deterministic algorithm.

## Status

🌱 **v1 pipeline working.** The deterministic A0→B→C pipeline idealizes the
Daikonic mark end-to-end (SSIM 0.98 / mean ΔE 0.006 vs. source) into a 1.9 KB
structured SVG with exact bilateral symmetry. See the design spec and plan:
- spec: [`docs/superpowers/specs/2026-06-04-vectormark-logo-idealizer-design.md`](docs/superpowers/specs/2026-06-04-vectormark-logo-idealizer-design.md)
- plan: [`docs/superpowers/plans/2026-06-04-vectormark-v1.md`](docs/superpowers/plans/2026-06-04-vectormark-v1.md)

```bash
uv sync --extra dev
uv run vectormark path/to/logo.png -o out.svg
uv run pytest
```

## Pipeline (v1)

```
A0  color optimization   robust palette (OKLab, ΔE-merge); AA halo discarded as noise
B   trace + clean        vtracer per layer; drop background plate
C   idealize             symmetry axis (PCA + reflection gate) → fundamental domain
                         → contour → RDP → primitive-first / Bézier-fallback
                         → geometry-level mirror (<use>)
→   structured SVG       native shapes + <use>; optional --flatten to paths
```

v1 targets flat-color segmented marks. v1.1 adds gradient normalize/re-apply.

## Prototype → core

Built first in **Python** (numpy/scipy/scikit-image/shapely/Pillow + vtracer),
with a pure, IO-free pipeline so the eventual **Rust core** (npm via napi-rs,
PyPI via PyO3/maturin, WASM for browser) is a mechanical port.

## License

MIT
