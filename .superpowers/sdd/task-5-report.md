Task 5 Report: Pass 2a - primitives

Summary
- Added the new optimizer pass package at `src/vectormark/optimizer/passes/`.
- Implemented `primitives_pass(objects, masks, *, epsilon=1.5) -> list[Proposal]`.
- Added focused tests for faithful disk promotion, irregular-blob no-op behavior, deterministic MultiPolygon handling, and empty/non-polygon skips.

Implementation details
- `primitives_pass` processes objects in stable `id` order.
- For each object, it samples points from `obj.flat` by taking:
  - the polygon exterior for `Polygon`
  - the exterior of the largest non-empty polygon by area for `MultiPolygon`
- Empty and non-polygon geometries are skipped.
- The pass calls `recognize_primitive(points, epsilon=epsilon)`.
- A proposal is emitted only when:
  - a primitive is recognized, and
  - the recognized shape differs from `obj.exact`
- Replacement objects are created via `obj.with_exact(primitive)`.
- The pass remains additive and is not wired into `idealize()`.

Files changed
- `src/vectormark/optimizer/passes/__init__.py`
- `src/vectormark/optimizer/passes/primitives.py`
- `tests/optimizer/test_pass_primitives.py`

TDD flow
1. Added `tests/optimizer/test_pass_primitives.py` first.
2. Confirmed the expected red state (`ModuleNotFoundError` for the new pass package).
3. Implemented the pass package and `primitives_pass`.
4. Re-ran the focused test target to green.

Test results
- Passed: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_pass_primitives.py -q`
- Not run: `tests/optimizer/test_framework.py` because framework code and imports were not changed.

Notes
- The pass currently relies on the object's flattened geometry, not the original path commands, which matches the task brief.
- The MultiPolygon selection uses an area-first tie-break with bounds as a deterministic secondary key.
- No changes were made outside the expected write scope.
