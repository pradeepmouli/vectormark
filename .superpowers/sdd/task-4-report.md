Task 4 Report: Optimizer framework

Status
- Complete.

Files changed
- `src/vectormark/optimizer/framework.py`
- `tests/optimizer/test_framework.py`

Implementation summary
- Added `Proposal = namedtuple("Proposal", "obj_ids new_objects")`.
- Added `Pass` protocol with the required callable signature:
  `pass_fn(objects: list[OptObject], masks: dict[int, np.ndarray]) -> list[Proposal]`.
- Added `optimize(objects, masks, passes, *, budget=BUDGET) -> list[OptObject]`.

Optimizer behavior implemented
- Runs passes in the provided order.
- Collects each pass's proposals and applies them deterministically in sorted `obj_ids` order.
- Skips proposals deterministically when they reference ids already consumed earlier in the same pass.
- Rejects proposals if any referenced consumed id or mask is missing.
- Gates geometry-changing replacements against the matching consumed object's mask using
  `coverage_residual(new.flat, masks[orig_id]) <= budget`.
- Preserves masks for replacement ids.
- Assigns unioned consumed masks to new ids introduced by an accepted proposal.
- Leaves the framework additive only; no wiring into `idealize()`.

TDD notes
- Wrote the required failing framework test first.
- Confirmed the test failed during collection before implementation because
  `vectormark.optimizer.framework` did not yet exist.
- Implemented the framework and then added a focused multi-id test to lock down
  deterministic ordering, overlap skipping, and mask union behavior.

Focused verification
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_framework.py -q`
  - Result: `2 passed`
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_optobject.py tests/optimizer/test_gate.py -q`
  - Result: `9 passed`

Commit
- `be3a797` `feat(optimizer): pass framework with per-change coverage gate`

Concerns
- The task brief is explicit about gating replacement objects against matching consumed masks and about unioning masks for new ids. It is not explicit about whether a geometry-changing multi-id proposal that emits only brand-new ids should also be coverage-gated against a union mask. This implementation follows the brief literally: matched replacements are gated; new ids receive unioned masks for later passes.

Fix update
- Resolved the remaining concern about ungated new-id geometry from multi-id proposals.
- Updated `optimize()` so a replacement that cannot be matched to a single consumed original is gated against the union of the consumed masks, while the existing matching-id/single-id behavior stays unchanged.
- Accepted unmatched new ids still inherit that same consumed-union mask, preserving deterministic proposal ordering and overlap skipping.
- Added a regression test that rejects a two-id replacement whose brand-new rect covers only a tiny part of the consumed union.
- Added a positive test that accepts a brand-new rect when its geometry matches the consumed union mask closely enough, and verifies the union mask is assigned to the new id.

Final verification
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_framework.py -q`
  - Result:
    `....                                                                     [100%]`

Concern status
- Previous concern resolved: multi-id proposals can no longer introduce brand-new geometry without a coverage gate.

Task 4 aliasing fix
- Tightened replacement matching so an emitted id is only treated as a matched original when that id is part of the proposal's consumed `obj_ids`.
- Rejected proposals before mutation when any replacement id aliases a live object id that is still present in `current_objects` but is not consumed by that proposal.
- Added a regression with starting ids `1, 2, 3` that proposes `Proposal((1, 2), [replacement_with_id_3])` and verifies a later pass still sees unchanged ids and masks for `1`, `2`, and `3`.

Final test output
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_framework.py -q`
  - Result:
    `.....                                                                    [100%]`

Task 4 deterministic replacement ordering fix
- Sorted accepted replacement objects by stable `(id, z)` before reinserting them and before assigning replacement masks, so later passes no longer observe pass-provided replacement order.
- Added a regression where a proposal emits ids `8, 7` and a later pass observes canonical `[7, 8]`.
- Added a regression proving a matching-id replacement in a multi-id proposal is gated against its own original mask, not the consumed-mask union.

Final local test output
- `PYTHONPATH=src ./.venv/bin/python -m pytest tests/optimizer/test_framework.py -q`
  - Result:
    `.......                                                                  [100%]`
