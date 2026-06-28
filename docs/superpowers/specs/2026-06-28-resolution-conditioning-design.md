# Resolution Conditioning — Design

## Problem

vectormark's corpus is essentially noise-free (19/20 logos measure flat-area pixel-noise
0.000 — clean vector exports). The pipeline (segmentation, grammar, AA-coverage, scorer) was
built and validated on that distribution. **Out-of-distribution inputs** — AI-generated or
re-encoded rasters carrying generation/compression noise at high resolution — break it: the
V-bird (flat-noise 0.47 @ 1254×1254, ~2.5× the ~500px corpus norm) produces near-twin
quantized colors, spurious interior contour loops (holed mis-detection), fragmented dots, and
merged-region staircase.

**The mechanism is resolution × noise, not the noise alone.** Downscaling the V-bird to 512px
leaves its flat-noise metric ≈ 0.47 *unchanged* yet fixes the symptoms (holed nodes 2→0,
interior gaps 3.73%→0.51%, round dots) — because at lower resolution the same noise yields
fewer absolute boundary pixels, so jitter is small relative to feature size and stops spawning
spurious contour loops. It is a **feature-to-pixel-scale** effect.

## Goal

Normalize oversized inputs to a working resolution **before segmentation**, turning a dirty
high-res raster into a corpus-like input. This is the robust fix for the noisy-input class and
also cuts pipeline cost (fewer pixels). It is **resolution normalization only — NOT denoise**
(denoise is lossy guessing that smears real gradients; a good downscale filter suppresses
high-frequency noise as a near-lossless byproduct, subsuming it).

## Non-goals

- No denoise / median / bilateral filtering step. (Explicitly rejected: noise is a source/format
  artifact; the right response is conditioning or flagging, not lossy blurring.)
- No upscaling of small inputs (would invent detail).
- No change to segmentation, fitting, coverage, or scoring logic.

## Approach

A single conditioning step in `idealize()` (and `_idealize_rectified`), applied to the
flattened RGB array before `_segment_image`:

```
if max(H, W) > opt.working_max_dim:
    scale = opt.working_max_dim / max(H, W)
    new = (round(W*scale), round(H*scale))
    arr = LANCZOS-resize(arr, new)
```

- **Trigger / target:** downscale only when the longest side exceeds `working_max_dim`;
  resize so the longest side equals `working_max_dim`, aspect-preserving. Inputs already
  within the threshold pass through untouched (the entire ≤500px clean corpus is unaffected).
- **Filter:** PIL `LANCZOS` (high-quality, deterministic, anti-aliasing) for the downscale.
- **`working_max_dim` default = 768.** Rationale: the V-bird sweep showed holed=0 at both 512
  and 768; 512 gave the lowest residual gap but 768 retains more legitimate detail for
  genuinely-detailed marks. 768 is the conservative default; tunable via `Options`. (512 noted
  as the aggressive setting validated specifically on the V-bird.)
- **Default-on**, with an escape hatch: `Options.working_max_dim: int | None = 768`; set to
  `None` to disable conditioning entirely (pass-through).

## Coordinate space / output dimensions

Segmentation and fitting run in the **conditioned** coordinate space. To avoid surprising a
caller who passed a 1254px image, the emitted SVG preserves the **original** pixel dimensions
as the document `width`/`height`, while the `viewBox` stays in conditioned space — a pure
display scale, zero coordinate surgery (SVG is resolution-independent, so this renders
identically and at the original size). Concretely: `render_svg_doc` already emits
`viewBox="0 0 {w} {h}"`; add original `width`/`height` attributes when conditioning occurred.

(Alternative considered: scale every fitted coordinate back to original space via a uniform
affine. Rejected as unnecessary coordinate churn — the display-scale approach is exact and
simpler. The `bake`/affine path is untouched.)

## Determinism

LANCZOS resize is deterministic for fixed input + size. No RNG. Same bytes + same
`working_max_dim` → byte-identical SVG.

## Interaction with the MCP server

The MCP server already exposes `PreprocessOpts` (PR #39). Conditioning lives in the **core**
`idealize()` so it is automatic for every entry point (CLI, library, MCP). The MCP
`PreprocessOpts` may later surface `working_max_dim` as a tunable, but that is out of scope
here — the default-on core behavior is the deliverable.

## Validation

- **Determinism test:** conditioned idealize is byte-identical across two runs.
- **Pass-through test:** an input with `max(H,W) ≤ working_max_dim` is untouched — byte-identical
  SVG to `working_max_dim=None`.
- **Downscale test:** a 1500px synthetic input is segmented at ≤768px (assert internal working
  dims), output `width`/`height` = original.
- **Corpus regression:** the synthetic golden tests (all ≤500px) are unaffected. Real-logo
  smoke: the V-bird improves (holed nodes → 0, a node emits `<circle>`); the larger corpus
  members (burger_king 1280×1395, contact_sheet 474×2822) must not lose shapes or gain speckle —
  re-derive any real-logo reference only if structure is preserved, else STOP and report.

## Open question (resolved at default)

`working_max_dim` exact value is a tuning parameter; 768 chosen as the conservative default.
If corpus validation shows a large clean member (e.g. burger_king) regressing at 768, raise the
default so that member passes through untouched, and document the floor.
