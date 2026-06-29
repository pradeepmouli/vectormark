Task 2 report: Stage-1 faithful vectorize -> objects + true masks

Scope
- Added `src/vectormark/optimizer/faithful.py`.
- Added `tests/optimizer/test_faithful.py`.
- Kept the change additive. No wiring into `idealize()` and no existing pipeline output changes.

What changed
- Implemented `faithful_objects(arr, opt) -> (objects, true_masks)`.
- Reused the current Stage-1 front-end pieces:
  - `_segment_image()` for quantize/segment and optional AA coverage attachment.
  - `decompose_components()` so merge behavior stays component-local like `_render_body`.
  - `merge_surfaces()` on per-component flat-filled regions.
  - `fit_fill()` on the merged region masks, matching the current pipeline's post-merge fill decision.
  - `region_contours()` + `fit_path()` for faithful geometry generation only.
- Built deterministic `OptObject` output:
  - order sorted explicitly by area descending, then `label`, then `color_hex`
  - `id` and `z` both assigned from that stable ordering
  - `true_masks[obj.id]` stored as the merged region responsibility mask
- Added hole support for faithful paths:
  - all significant contours are fit independently
  - multi-contour shapes are emitted as one `Shape("path", {"d": ..., "fill_rule": "evenodd"})`

Tests added
- `test_faithful_single_disk_one_object_path`
  - asserts one object
  - asserts faithful geometry stays a `path`, not a primitive
  - asserts a true mask exists
  - asserts flattened area is within 10 percent of the raster disk area
- `test_faithful_gradient_strip_merges_to_one_gradient_object`
  - constructs a horizontal two-color strip
  - asserts Stage 1 surface merging yields one object
  - asserts the merged fill is gradient-typed
  - asserts the stored true mask matches the merged strip area

Verification
- Failing-first check:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_faithful.py -q`
  - initial failure: `ModuleNotFoundError: No module named 'vectormark.optimizer.faithful'`
- Passing checks:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_faithful.py -q`
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_optobject.py tests/optimizer/test_faithful.py -q`

Notes
- The implementation filters multi-contour holes by a significance threshold derived from `opt.min_region_fraction`, with a 1-pixel absolute floor, then falls back to the outer contour if needed. This keeps tiny contour noise out of Stage 1 while preserving meaningful holes.
- No symmetry, primitive recognition, `recognize_primitive`, `select_geometry`, or `symmetry.py` code is called.

Result
- Task 2 is implemented and verified in focused optimizer tests.
