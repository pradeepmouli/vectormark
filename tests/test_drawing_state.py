from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from vectormark.drawing_state import DrawingNotFound, DrawingStore
from vectormark.drawing_trace import TraceOptions, TraceResult


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


def _scene() -> object:
    return object()


def test_store_creates_immutable_root_version(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    drawing = store.create(object(), _trace())

    assert drawing.id.startswith("drw_")
    assert drawing.versions == {
        "v0": drawing.versions["v0"],
    }
    assert drawing.versions["v0"].id == "v0"
    assert drawing.versions["v0"].parent_id is None
    assert drawing.versions["v0"].plan is None
    assert drawing.versions["v0"].scene is None
    assert drawing.versions["v0"].label is None


def test_store_branches_from_any_retained_version(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace())

    first = store.append(session, drawing.id, "v0", plan={}, scene=_scene())
    second = store.append(session, drawing.id, "v0", plan={}, scene=_scene())
    child = store.append(session, drawing.id, "v0.1", plan={}, scene=_scene())

    assert (first.id, second.id, child.id) == ("v0.0", "v0.1", "v0.1.0")
    assert child.parent_id == "v0.1"


def test_store_allocates_sibling_versions_atomically(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock)
    session = object()
    drawing = store.create(session, _trace())

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = list(
            executor.map(
                lambda _: store.append(session, drawing.id, "v0", plan={}, scene=_scene()).id,
                range(32),
            )
        )

    assert sorted(versions, key=lambda version: int(version.rsplit(".", 1)[1])) == [
        f"v0.{index}" for index in range(32)
    ]


def test_store_rejects_cross_session_and_expired_drawings(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    owner, other = object(), object()
    drawing = store.create(owner, _trace())

    with pytest.raises(DrawingNotFound):
        store.get(other, drawing.id, "v0")
    fake_clock.advance(1801)
    with pytest.raises(DrawingNotFound):
        store.get(owner, drawing.id, "v0")


def test_store_get_and_append_refresh_the_sliding_expiry(fake_clock: FakeClock) -> None:
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    session = object()
    drawing = store.create(session, _trace())

    fake_clock.advance(1799)
    state, root = store.get(session, drawing.id, "v0")
    assert state is drawing
    assert root.id == "v0"

    fake_clock.advance(1799)
    child = store.append(session, drawing.id, "v0", plan={}, scene=_scene(), label="first edit")
    assert child.label == "first edit"

    fake_clock.advance(1799)
    _, retained_child = store.get(session, drawing.id, child.id)
    assert retained_child is child

