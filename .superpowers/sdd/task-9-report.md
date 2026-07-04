# Task 9 Report: Optimizer Integration

## Summary

- Added `Options.optimizer: bool = False`.
- Wired an experimental optimizer branch in `idealize()` behind that option.
- The branch runs `faithful_objects(arr, opt)` -> `optimize(..., [primitives_pass, clones_pass, symmetry_pass, simplify_pass])` -> SVG serialization.
- Serialization resolves optimizer object-id based `<use>` references to emitted SVG ids and uses `resolve_fill()` so gradients/raster fills still emit paint servers.
- Exported `clones_pass` from `vectormark.optimizer.passes`.
- Preserved native primitives during symmetry self-reconstruction so a disk optimized to `<circle>` is not rewritten back into a path.

## Tests

- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_integration.py tests/optimizer/test_pass_symmetry.py -q`
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer -q`
- `PYTHONPATH=src ./.venv/bin/python -m pytest -q --ignore=tests/test_mcp_server.py --ignore=tests/test_mcp_image.py --ignore=tests/test_candidate_byte_identical.py`

## Full-suite caveat

- The plan's full non-MCP command still fails in `tests/test_candidate_byte_identical.py` for:
  - `daikonic`
  - `daikonic_flatten`
  - `rectified_grad`
  - `rectified_grad_flatten`
- Those same four failures reproduce at pre-Task-9 commit `18ce1a3`, so they are not introduced by the optimizer integration.

## Coverage

- Default `Options()` output remains routed through the legacy pipeline.
- `Options(optimizer=True)` is deterministic on a disk image and emits `<circle>`.
- An asymmetric single cloud-like object remains a path and does not emit `<use>`.
- The real Daikonic fixture produces a non-empty multi-element optimizer SVG with the full source image, including the wordmark region.
