# Task 1 Report: OptObject model + flatten

Status: complete

Files changed:
- `/Users/pmouli/GitHub.nosync/active/py/vectormark/src/vectormark/optimizer/__init__.py`
- `/Users/pmouli/GitHub.nosync/active/py/vectormark/src/vectormark/optimizer/optobject.py`
- `/Users/pmouli/GitHub.nosync/active/py/vectormark/tests/optimizer/test_optobject.py`

Summary:
- Added the new `vectormark.optimizer` package without wiring it into `pipeline.py`.
- Implemented `flatten_points(shape, samples=24)` by converting `Shape` to a path `d` via `shape_to_path_d` and sampling `M/L/Q/C/Z` segments.
- Implemented `to_polygon(shape, samples=24)` to build shapely polygon geometry from one or more sampled subpaths, treating the largest ring as shell and subtracting remaining rings as holes, with `.buffer(0)` fallback validation.
- Added frozen `OptObject` with cached `flat` geometry derived from `exact`, plus `with_exact(new_shape)` to return a refreshed copy.

TDD evidence:

RED:
- Command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_optobject.py -q`
- Result:
  - Failed during collection with `ModuleNotFoundError: No module named 'vectormark.optimizer'`

GREEN:
- Command:
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_optobject.py -q`
- Result:
  - `3 passed`

Notes:
- The implementation follows the exact Task 1 scope and stays additive.
- No pipeline integration was added.
- Unrelated untracked workspace files were left untouched.
