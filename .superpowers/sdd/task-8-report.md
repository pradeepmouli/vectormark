# Task 8 Report: Simplify Pass

## Summary

- Added `simplify_pass(objects, masks) -> list[Proposal]` for path-only geometry simplification.
- The pass samples the current path geometry, rebuilds it through the existing `fit_path` RDP/curve fitting helper, and proposes the replacement only when SVG command count decreases.
- Simplification is restricted to single-shell polygonal paths so multipart and holed geometry are not accidentally collapsed by dominant-ring sampling.
- The pass is exported from `vectormark.optimizer.passes`; pipeline integration is intentionally left for Task 9.

## Tests

- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_pass_simplify.py -q`
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer -q`

## Coverage

- Long straight `L` runs collapse to fewer line segments.
- Smooth sampled arcs reduce to curve commands.
- `optimize` rejects an over-aggressive simplification that drops a real bump via the existing coverage gate.

## Concerns

- Holed and multipart path simplification is intentionally skipped in this task; supporting those safely would need per-ring reconstruction.
- The pass is not wired into pipeline integration yet; that remains Task 9.
