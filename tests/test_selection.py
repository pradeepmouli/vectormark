import pytest

from vectormark.selection import (
    ElementSelection, SelectionPolicy, validate_strategies,
    KNOWN_STRATEGIES, PRIMITIVE, PATH, SYMMETRIC,
)


def test_known_strategies_has_all_nine_labels():
    assert KNOWN_STRATEGIES == {
        "primitive", "trapezoid", "sym_polygon", "cap", "symmetric",
        "polygon", "path", "holed_symmetric", "holed_path",
    }


def test_for_id_returns_specific_then_default_then_none():
    sel = ElementSelection(force=PATH)
    deflt = ElementSelection(allow=frozenset({PRIMITIVE}))
    policy = SelectionPolicy(by_id={"s3": sel}, default=deflt)
    assert policy.for_id("s3") is sel        # specific wins
    assert policy.for_id("s9") is deflt       # falls back to default
    assert SelectionPolicy().for_id("s0") is None  # no entry, no default


def test_validate_accepts_known_labels():
    validate_strategies(ElementSelection(allow=frozenset({PRIMITIVE, PATH}), force=SYMMETRIC))  # should not raise


def test_validate_rejects_unknown_allow_label():
    with pytest.raises(ValueError, match="symetric"):
        validate_strategies(ElementSelection(allow=frozenset({"symetric"})))


def test_validate_rejects_unknown_force_label_even_when_allow_none():
    with pytest.raises(ValueError, match="blob"):
        validate_strategies(ElementSelection(force="blob"))


from vectormark.candidate import Candidate, FlatFill
from vectormark.fit import Shape


def test_candidate_strategy_defaults_none_and_is_settable():
    c0 = Candidate(Shape("circle", {}), FlatFill("#000000"), "region")
    assert c0.strategy is None                       # backward compatible default
    c1 = Candidate(Shape("path", {"d": "M0 0"}), FlatFill("#000000"),
                   "region", strategy="path")
    assert c1.strategy == "path"
