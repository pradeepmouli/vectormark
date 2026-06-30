Task 7 Report: Pass 2c - object-level symmetry

Status
- Implemented the object-level symmetry optimizer pass in `src/vectormark/optimizer/passes/symmetry.py`.
- Added focused tests in `tests/optimizer/test_pass_symmetry.py`.
- Exported `symmetry_pass` from `src/vectormark/optimizer/passes/__init__.py`.
- Did not wire the pass into `idealize()` or the optimizer pipeline; that remains Task 9.

What changed
- Added `symmetry_pass(objects, masks) -> list[Proposal]`.
- Mirror-pair proposals:
  - consider polygonal `flat` geometries only,
  - skip existing `Shape("use")` objects,
  - require `FlatFill` on both source and target for safe `<use>` fill overrides,
  - process objects in stable ascending id order,
  - use the lower-id object as the canonical source,
  - derive the mirror axis from the centroid perpendicular bisector,
  - verify area/perimeter similarity, geometric reflection residual, and `gate_ok()` against the target mask before proposing,
  - replace only the mirrored target object with `Shape("use", {"href_obj_id": canonical.id, "transform": matrix, "fill": target_fill_hex})`,
  - keep the replacement `flat` as the reflected canonical geometry.
- Self-symmetry proposals:
  - test centroid axes derived from the minimum rotated rectangle plus horizontal/vertical candidates,
  - verify reflected geometry residual before reconstruction,
  - reconstruct from one clipped half plus its exact reflection across the selected axis,
  - serialize the reconstructed geometry as a `Shape("path")`,
  - gate the reconstructed geometry against the original mask before proposing.
- Pair detection runs before self-symmetry so paired symmetric objects compress to a canonical object plus `<use>` instead of being rewritten independently.

Test coverage added
- mirror-pair rectangles -> target becomes `<use>` with object-id href and reflection matrix
- self-symmetric diamond -> rewritten as a gated exact path reconstruction
- non-flat fill mirror pair -> no `<use>` proposal
- unordered inputs -> deterministic proposal output
- target mask mismatch -> mirror-pair proposal rejected by gate

Verification
- Command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer -q`
- Result:
  - `...........................................                              [100%]`
  - `43 passed`

Notes / concerns
- `<use>` mirror-pair proposals intentionally remain `FlatFill`-only for v1, matching the clone pass constraint.
- The pass is object-level only and is not integrated into the pipeline yet.
- Self-symmetry reconstruction emits path geometry even when a later task may choose to preserve or re-recognize primitives during full pipeline integration.
