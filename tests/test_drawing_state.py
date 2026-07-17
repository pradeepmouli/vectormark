from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass

import pytest
import numpy as np

from vectormark.candidate import FlatFill
from vectormark.drawing_state import DrawingNotFound, DrawingStore
from vectormark.drawing_trace import TraceOptions, TracePath, TraceRegion, TraceResult
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def _trace() -> TraceResult:
    return TraceResult(
        width=1,
        height=1,
        options=TraceOptions(),
        regions=(),
        region_map_svg="<svg/>",
    )


def _regions() -> tuple[VectorRegion, ...]:
    return (
        VectorRegion.from_shape(
            id=1,
            shape=Shape("path", {"d": "M0 0Z"}),
            fill=FlatFill("#112233"),
            z=0,
            raster=np.ones((1, 1), dtype=bool),
            drawing_id="r1",
            source_regions=("r1",),
        ),
    )


def _mutable_trace() -> TraceResult:
    mask = np.array([[True, False], [False, False]])
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return TraceResult(
        width=2,
        height=2,
        options=TraceOptions(),
        regions=(
            TraceRegion(
                id="r1",
                source_label=1,
                color="#ffffff",
                mask=mask,
                contours=(contour,),
                trace_path=TracePath(d="M0 0Z", fill_rule="nonzero", commands=()),
                effective_trace_level="pixel",
            ),
        ),
        region_map_svg="<svg/>",
    )


def test_store_creates_immutable_root_version(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    drawing = store.create(object(), _trace(), regions=_regions())

    assert drawing.id.startswith("drw_")
    assert drawing.versions == {
        "v0": drawing.versions["v0"],
    }
    assert drawing.versions["v0"].id == "v0"
    assert drawing.versions["v0"].parent_id is None
    assert drawing.versions["v0"].plan is None
    assert drawing.versions["v0"].regions is not None
    assert drawing.versions["v0"].label is None
    with pytest.raises(TypeError):
        drawing.versions["injected"] = drawing.versions["v0"]
    with pytest.raises(FrozenInstanceError):
        drawing.versions = {}


def test_store_detaches_plan_and_vector_regions(
    fake_clock: FakeClock,
) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())
    plan = {"palette": ["blue"], "limits": {"strokes": 2}}
    regions = _regions()

    version = store.append(session, drawing.id, "v0", plan=plan, regions=regions)
    plan["palette"].append("red")
    plan["limits"]["strokes"] = 3
    regions[0].raster[0, 0] = False

    assert version.plan == {"palette": ("blue",), "limits": {"strokes": 2}}
    assert version.regions is not None
    assert version.regions[0].raster[0, 0]
    with pytest.raises(TypeError):
        version.plan["injected"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        version.plan["palette"].append("green")  # type: ignore[union-attr]
    assert version.regions[0].drawing_id == "r1"

    _, stored = store.get(session, drawing.id, version.id)
    assert stored.plan == {"palette": ("blue",), "limits": {"strokes": 2}}
    assert stored.regions is not None
    assert stored.regions[0].raster[0, 0]


def test_store_rejects_non_vector_region_state(
    fake_clock: FakeClock,
) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())
    with pytest.raises(TypeError, match="regions"):
        store.append(
            session,
            drawing.id,
            "v0",
            plan={"palette": ["blue"]},
            regions=object(),  # type: ignore[arg-type]
        )
    _, root = store.get(session, drawing.id, "v0")
    assert root.regions is not None


def test_store_branches_from_any_retained_version(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())

    first = store.append(session, drawing.id, "v0", plan={}, regions=_regions())
    second = store.append(session, drawing.id, "v0", plan={}, regions=_regions())
    child = store.append(session, drawing.id, "v0.1", plan={}, regions=_regions())

    assert (first.id, second.id, child.id) == ("v0.0", "v0.1", "v0.1.0")
    assert child.parent_id == "v0.1"


def test_store_allocates_sibling_versions_atomically(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = list(
            executor.map(
                lambda _: store.append(session, drawing.id, "v0", plan={}, regions=_regions()).id,
                range(32),
            )
        )

    assert sorted(versions, key=lambda version: int(version.rsplit(".", 1)[1])) == [
        f"v0.{index}" for index in range(32)
    ]


def test_store_accepts_request_session_handoffs_and_expires_drawings(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    owner, other = object(), object()
    drawing = store.create(owner, _trace(), regions=_regions())

    # An outbound MCP tunnel can create a fresh transport session for every
    # request.  The opaque drawing ID, rather than the transient session
    # object, is the live-state capability.
    _, root = store.get(other, drawing.id, "v0")
    assert root.id == "v0"

    fake_clock.advance(1800)
    with pytest.raises(DrawingNotFound) as expired_error:
        store.get(other, drawing.id, "v0")
    assert expired_error.value.__cause__ is None


def test_store_not_found_errors_do_not_expose_internal_key_errors(
    fake_clock: FakeClock,
) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())

    with pytest.raises(DrawingNotFound) as missing_version_error:
        store.get(session, drawing.id, "v404")
    assert missing_version_error.value.__cause__ is None

    with pytest.raises(DrawingNotFound) as missing_parent_error:
        store.append(session, drawing.id, "v404", plan={}, regions=_regions())
    assert missing_parent_error.value.__cause__ is None


def test_store_get_and_append_refresh_the_sliding_expiry(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())

    fake_clock.advance(1799)
    state, root = store.get(session, drawing.id, "v0")
    assert state == drawing
    assert root.id == "v0"

    fake_clock.advance(1799)
    child = store.append(session, drawing.id, "v0", plan={}, regions=_regions(), label="first edit")
    assert child.label == "first edit"

    fake_clock.advance(1799)
    _, retained_child = store.get(session, drawing.id, child.id)
    assert retained_child.id == child.id
    assert retained_child.label == child.label


def test_store_rejects_non_exact_string_labels_without_consuming_child_number(
    fake_clock: FakeClock,
) -> None:
    class StringSubclass(str):
        pass

    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace(), regions=_regions())

    for label in (StringSubclass("subclass"), object()):
        with pytest.raises(TypeError, match="label"):
            store.append(session, drawing.id, "v0", plan={}, regions=_regions(), label=label)

    version = store.append(session, drawing.id, "v0", plan={}, regions=_regions(), label="valid")
    assert version.id == "v0.0"


def test_store_detaches_trace_arrays_from_inputs_and_public_snapshots(
    fake_clock: FakeClock,
) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    trace = _mutable_trace()
    source_region = trace.regions[0]
    source_region.mask.setflags(write=False)
    source_region.contours[0].setflags(write=False)

    created = store.create(session, trace, regions=_regions())
    source_region.mask.setflags(write=True)
    source_region.contours[0].setflags(write=True)
    source_region.mask[0, 0] = False
    source_region.contours[0][0, 0] = 99.0

    created_region = created.trace.regions[0]
    assert not created_region.mask.flags.writeable
    assert not created_region.contours[0].flags.writeable
    created_region.mask.setflags(write=True)
    created_region.mask[0, 0] = False

    later, _ = store.get(session, created.id, "v0")
    later_region = later.trace.regions[0]
    assert later_region.mask[0, 0]
    assert later_region.contours[0][0, 0] == 0.0
    assert not later_region.mask.flags.writeable
    assert not later_region.contours[0].flags.writeable
