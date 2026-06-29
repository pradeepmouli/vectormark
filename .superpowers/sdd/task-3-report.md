Task 3 report: Coverage gate for the SVG Geometry Optimizer

Scope
- Added `src/vectormark/optimizer/gate.py`.
- Added `tests/optimizer/test_gate.py`.
- Kept the change additive. No wiring into `idealize()` and no changes to earlier optimizer stages.

What changed
- Implemented the required coverage gate interface:
  - `BUDGET = 0.02`
  - `rasterize(geom, shape_hw) -> np.ndarray`
  - `coverage_residual(geom, true_mask) -> float`
  - `gate_ok(geom, true_mask, *, budget=BUDGET) -> bool`
- Implemented mask rasterization for:
  - `shapely.geometry.Polygon`
  - `shapely.geometry.MultiPolygon`
- Used `skimage.draw.polygon` exactly as specified:
  - fill the polygon exterior
  - rasterize each interior ring
  - subtract hole masks from the exterior mask
  - union masks across `MultiPolygon` members
- Implemented residual scoring as normalized symmetric difference against the provided `true_mask`:
  - `xor = rasterize(geom, true_mask.shape) XOR true_mask`
  - `coverage_residual = xor.sum() / true_mask.sum()`
- Added defensive handling for an empty `true_mask`:
  - returns `0.0` when both masks are empty
  - returns `1.0` when the truth mask is empty but the candidate raster is not
  - this avoids divide-by-zero while preserving gate rejection for a false positive candidate

Tests added
- `test_gate_accepts_matching_circle`
  - builds a disk mask
  - compares it to a high-resolution buffered circle
  - asserts residual stays below `0.05`
  - asserts the default gate accepts it
- `test_gate_rejects_wrong_shape`
  - compares the same disk mask to a covering square
  - asserts residual exceeds `BUDGET`
  - asserts the default gate rejects it

Verification
- Focused test command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_gate.py -q`
- Result:
  - `..                                                                       [100%]`

Result
- Task 3 is implemented and verified in the requested focused optimizer tests.

Fix after review
- Locked the budget contract in tests with an explicit `assert BUDGET == 0.02`.
- Tightened the default gate contract by asserting a mismatched square is rejected at the default budget and accepted when the budget is raised just above its measured residual.
- Added a deterministic polygon-with-hole case:
  - builds a square ring with an interior square hole
  - asserts `rasterize()` subtracts the hole from the filled exterior
  - asserts `coverage_residual(...) == 0.0`
  - asserts the default gate accepts the matching mask
- Added a deterministic `MultiPolygon` case:
  - builds two disjoint squares
  - asserts `rasterize()` fills both components
  - asserts `coverage_residual(...) == 0.0`
  - asserts the default gate accepts the matching mask

Final verification
- Focused test command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_gate.py -q`
- Result:
  - `....                                                                     [100%]`
