import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.data_loading import ACTIVITY, CASE_ID, TIMESTAMP
from src.features import build_feature_table, build_preprocessor, extract_prefix_features
from src.prefixes import EVENTS, REMAINING_TIME, extract_prefixes


def test_extract_prefix_features_for_two_events():
    events = [
        {
            ACTIVITY: "Register",
            TIMESTAMP: pd.Timestamp("2020-01-01 00:00"),
            "customer": "ann",
        },
        {
            ACTIVITY: "Check",
            TIMESTAMP: pd.Timestamp("2020-01-02 00:00"),
            "customer": "ann",
        },
    ]

    features = extract_prefix_features(events, static_attributes=["customer"])

    assert features["customer"] == "ann"
    assert features["prefix_length"] == 2
    assert features["elapsed_time"] == 24
    assert features["time_since_previous"] == 24
    assert features["last_activity"] == "Check"
    assert features["activity_counts"] == {"Register": 1, "Check": 1}


def event(activity="Register", **attributes):
    return {ACTIVITY: activity, TIMESTAMP: pd.Timestamp("2020-01-06"), **attributes}


def raw_table(*prefixes, static_attributes=()):
    return build_feature_table(
        pd.DataFrame({EVENTS: list(prefixes)}),
        static_attributes=list(static_attributes),
    )


def encoded_table(preprocessor, raw):
    matrix = preprocessor.transform(raw)
    values = matrix.toarray() if issparse(matrix) else matrix
    return pd.DataFrame(values, columns=preprocessor.get_feature_names_out())


def test_missing_values_use_training_median_and_mode():
    fields = ["customer", "cost"]
    train = raw_table(
        [event(customer="ann", cost=2)],
        [event(customer="bob", cost=10)],
        [event(customer="ann", cost=12)],
        static_attributes=fields,
    )
    new = raw_table([event(customer=None, cost=np.nan)], static_attributes=fields)
    preprocessor = build_preprocessor(
        categorical_attributes=["customer"], numeric_attributes=["cost"], scale_numeric=False
    ).fit(train)

    result = encoded_table(preprocessor, new)
    assert result.loc[0, "numeric__cost"] == 10
    assert result.loc[0, "category__customer_ann"] == 1
    assert result.loc[0, "category__customer_bob"] == 0
    assert np.isfinite(result.to_numpy()).all()


def test_absent_configured_field_raises_key_error():
    with pytest.raises(KeyError, match="customer"):
        extract_prefix_features([event()], static_attributes=["customer"])


def test_unseen_activity_customer_and_extra_column():
    train = raw_table([event(customer="ann")], static_attributes=["customer"])
    new = raw_table(
        [event(customer="new"), event("Escalate", customer="new")],
        static_attributes=["customer"],
    )
    preprocessor = build_preprocessor(
        categorical_attributes=["customer"], scale_numeric=False
    ).fit(train)

    expected_columns = encoded_table(preprocessor, train).columns
    result = encoded_table(preprocessor, new.assign(extra_feature=123))
    assert result.columns.equals(expected_columns)
    assert "count__Escalate" not in result.columns
    assert result.loc[0, "numeric__prefix_length"] == 2
    assert result.loc[0, "count__Register"] == 1
    assert result.loc[0, "category__last_activity_Register"] == 0
    assert result.loc[0, "category__customer_ann"] == 0
    pd.testing.assert_frame_equal(result, encoded_table(preprocessor, new))


def test_unsupported_time_unit():
    with pytest.raises(ValueError, match="time_unit"):
        extract_prefix_features([event()], static_attributes=[], time_unit="minutes")


def test_single_event_prefix():
    features = extract_prefix_features(
        [event(owner="team A")], static_attributes=[], reassignment_field="owner"
    )
    assert features["prefix_length"] == 1
    assert features["elapsed_time"] == 0
    assert features["time_since_previous"] == 0
    assert features["last_activity"] == "Register"
    assert features["activity_counts"] == {"Register": 1}
    assert features["reassignment_count"] == 0


def test_changing_future_event_preserves_prefix_features():
    log = pd.DataFrame({
        CASE_ID: ["c1"] * 3,
        ACTIVITY: ["Register", "Check", "Approve"],
        TIMESTAMP: pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]),
    })
    changed = log.copy()
    changed.loc[2, ACTIVITY] = "Reject"
    changed.loc[2, TIMESTAMP] = pd.Timestamp("2020-01-05")

    original_prefixes = extract_prefixes(log)
    changed_prefixes = extract_prefixes(changed)
    assert original_prefixes[REMAINING_TIME].tolist() == [72, 48]
    assert changed_prefixes[REMAINING_TIME].tolist() == [96, 72]

    original = build_feature_table(original_prefixes, static_attributes=[])
    updated = build_feature_table(changed_prefixes, static_attributes=[])
    pd.testing.assert_frame_equal(original, updated)


@pytest.mark.parametrize(
    ("unit", "elapsed", "previous"), [("hours", 30, 6), ("days", 1.25, 0.25)]
)
def test_elapsed_and_previous_intervals(unit, elapsed, previous):
    start = pd.Timestamp("2020-01-06")
    events = [
        {ACTIVITY: "Check", TIMESTAMP: start + pd.Timedelta(hours=offset)}
        for offset in (0, 24, 30)
    ]
    features = extract_prefix_features(events, static_attributes=[], time_unit=unit)
    assert features["elapsed_time"] == elapsed
    assert features["time_since_previous"] == previous


def test_scaling_uses_training_statistics_only():
    first = event()
    day_two = {ACTIVITY: "Check", TIMESTAMP: first[TIMESTAMP] + pd.Timedelta(days=1)}
    day_three = {ACTIVITY: "Check", TIMESTAMP: first[TIMESTAMP] + pd.Timedelta(days=2)}
    train = raw_table([first], [first, day_two])
    validation = raw_table([first, day_three])
    preprocessor = build_preprocessor(scale_numeric=True).fit(train)

    column = "numeric__elapsed_time"
    before = encoded_table(preprocessor, train)
    np.testing.assert_allclose(before[column], [-1, 1])
    np.testing.assert_allclose(encoded_table(preprocessor, validation)[column], [3])
    pd.testing.assert_frame_equal(encoded_table(preprocessor, train), before)


def test_feature_rows_remain_aligned_with_targets():
    prefixes = pd.DataFrame(
        {EVENTS: [[event()], [event(), event("Check")]], REMAINING_TIME: [72, 48]},
        index=[20, 5],
    )
    raw = build_feature_table(prefixes, static_attributes=[])

    pd.testing.assert_index_equal(raw.index, prefixes.index)
    aligned = raw.join(prefixes[REMAINING_TIME])
    assert list(zip(aligned["last_activity"], aligned[REMAINING_TIME])) == [
        ("Register", 72), ("Check", 48)
    ]
