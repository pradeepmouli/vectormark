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

Fix follow-up
- What changed
  - Removed the Stage 1 contour significance filter from `src/vectormark/optimizer/faithful.py` so faithful path generation preserves every contour returned by `region_contours()`, including real inner holes.
  - Strengthened `tests/optimizer/test_faithful.py` with a regression for a small but meaningful hole that the old `largest * opt.min_region_fraction` cutoff dropped.
  - Reworked the gradient merge test to first prove `_segment_image()` produces at least two adjacent non-background regions, then assert `faithful_objects()` merges them into one gradient-filled object.
- RED/GREEN evidence for the new and strengthened tests
  - RED: old contour-threshold behavior on the small-hole case dropped the hole entirely:
    - Command:
      - `PYTHONPATH=src ./.venv/bin/python - <<'PY' ... emulate old contour cutoff on the small-hole case ... PY`
    - Output:
      - `{'contours': 2, 'kept_by_old_threshold': 1, 'min_area': 127.21, 'hole_polygon_area': 80.5, 'fill_rule': None, 'flat_area': 6374.22, 'mask_area': 6280}`
  - RED: the original gradient-strip setup with `max_colors=2` did not prove merging because segmentation produced only one non-background region before `faithful_objects()` ran:
    - Command:
      - `PYTHONPATH=src ./.venv/bin/python - <<'PY' ... compare max_colors=2 vs max_colors=3 for the strip case ... PY`
    - Output:
      - `{'max_colors': 2, 'premerge_regions': 1, 'adjacent_pair': False, 'output_objects': 1, 'fill_type': 'RadialGradientFill'}`
      - `{'max_colors': 3, 'premerge_regions': 2, 'adjacent_pair': True, 'output_objects': 1, 'fill_type': 'RadialGradientFill'}`
  - GREEN: the focused Task 2 pytest file now passes with the hole regression and strengthened merge proof in place.
- Final focused test run
  - Command:
    - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_faithful.py -q`
  - Output:
    - `...                                                                      [100%]`
