Task 6 Report: Pass 2b - clones

Status
- Implemented the clone-detection optimizer pass in `src/vectormark/optimizer/passes/clones.py`.
- Added focused tests in `tests/optimizer/test_pass_clones.py`.
- Extended `src/vectormark/emit.py` and `tests/test_emit.py` for `Shape("use")` serialization support.
- Did not wire the pass into `idealize()` yet, per task instructions.

What changed
- Added `clones_pass(objects, masks) -> list[Proposal]`.
- The pass:
  - considers polygonal `flat` geometries only,
  - requires flat fills for clone proposals,
  - processes objects in stable ascending id order,
  - uses the lower-id object as the canonical source,
  - buckets by simple invariant geometry descriptors (area, perimeter, part count),
  - verifies candidate congruence with a rigid rotation+translation transform derived from minimum-rotated-rectangle orientations plus centroid alignment,
  - confirms the transformed canonical geometry against the target mask before proposing a replacement.
- Clone replacements are emitted as:
  - `Shape("use", {"href": f"s{canonical_id}", "transform": (a, b, c, d, e, f), "fill": fill_hex})`
  - with an explicit transformed `flat` geometry passed into `OptObject(...)`.
- Extended `shape_to_svg()` to emit:
  - `<use id="..." href="#..." transform="matrix(...)" fill="..."/>`
- Extended `shape_to_path_d()` for `use` to:
  - transform `params["d"]` if present,
  - otherwise raise `ValueError("cannot convert use shape to path data without source geometry")`.

Test coverage added
- translated identical squares with different flat fills -> clone proposal accepted through `optimize()`
- rotated congruent square -> clone proposal accepted through `optimize()`
- square + circle -> no proposal
- non-flat fill clone target -> skipped
- `Shape("use")` SVG emission
- `Shape("use")` path conversion failure without source geometry

Verification
- Command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_pass_clones.py tests/test_emit.py -q`
- Result:
  - `18 passed`

Notes / concerns
- This v1 clone matcher is deliberately pragmatic: descriptor bucketing is lightweight, and transform verification relies on minimum-rotated-rectangle orientation candidates plus geometric residual and gate checks.
- Non-flat fill clone proposals are intentionally skipped for now because `Shape("use")` fill override support is only safe for `FlatFill` in this pass.

Fix after review
- Changed clone replacements to store `href_obj_id` instead of assuming `OptObject.id` equals emitted SVG id.
- Added `emit.resolve_use_shape()` and `emit.optimizer_objects_to_svg()` so optimizer object lists resolve object-id references to actual emitted ids before calling `shape_to_svg()`.
- Made direct `shape_to_svg(Shape("use", {"href_obj_id": ...}))` fail fast until the reference is resolved, while preserving literal `href` support.
- Added regression coverage for non-sequential object ids (`10`, `20`) serializing as emitted ids `s0`, `s1` with the clone using `href="#s0"`.

Final verification
- Command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_pass_clones.py tests/test_emit.py -q`
- Result:
  - `......................                                                   [100%]`
