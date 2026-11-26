"""Tests that use the synthetic log in tests/data/synthetic.yaml.

The log has four cases. They are far apart in time on purpose, so a chronological
split by case start time has one obvious answer:

    case  customer  events (day of January/February)                    duration
    c1    ann       Register 1 Jan, Check 2 Jan, Approve 4 Jan          3 days
    c2    bob       Register 10 Jan, Check 11 Jan, Reject 12 Jan        2 days
    c3    ann       Register 20 Jan, Check 22 Jan, Approve 23 Jan,
                    Archive 25 Jan                                      5 days
    c4    cy        Register 1 Feb, Check 2 Feb, Approve 5 Feb          4 days
"""

import pandas as pd
import pytest

from src.data_loading import ACTIVITY, CASE_ID, TIMESTAMP, RESOURCE
from src.explore import compute_statistics


def activities(log, case):
    return log.loc[log[CASE_ID] == case, ACTIVITY].tolist()


def test_synthetic_log_has_the_standard_columns(synthetic_log):
    # "Variant" in the CSV describes the whole case and must not be kept
    assert list(synthetic_log.columns) == [CASE_ID, ACTIVITY, TIMESTAMP, RESOURCE, "customer"]
    assert pd.api.types.is_datetime64_any_dtype(synthetic_log[TIMESTAMP])


def test_synthetic_log_is_sorted_by_case_and_time(synthetic_log):
    assert synthetic_log[CASE_ID].tolist() == ["c1"] * 3 + ["c2"] * 3 + ["c3"] * 4 + [
        "c4"
    ] * 3
    assert activities(synthetic_log, "c1") == ["Register", "Check", "Approve"]
    assert activities(synthetic_log, "c3") == [
        "Register",
        "Check",
        "Approve",
        "Archive",
    ]
    in_order = synthetic_log.groupby(CASE_ID)[TIMESTAMP].apply(
        lambda times: times.is_monotonic_increasing
    )
    assert in_order.all()


def test_static_attribute_is_constant_within_each_case(synthetic_log):
    assert (synthetic_log.groupby(CASE_ID)["customer"].nunique() == 1).all()


def test_synthetic_log_statistics(synthetic_log):
    stats = compute_statistics(synthetic_log)

    assert stats["cases"] == 4
    assert stats["events"] == 13
    assert stats["activities"] == 5
    assert stats["variants"] == 3  # c1 and c4 share Register-Check-Approve
    assert stats["case_duration_days"]["mean"] == pytest.approx(3.5)
    assert stats["case_duration_days"]["median"] == pytest.approx(3.5)
    assert stats["case_duration_days"]["max"] == pytest.approx(5.0)


def test_cases_do_not_overlap_in_time(synthetic_log):
    """A chronological split by case start has no case that straddles the cut."""
    spans = synthetic_log.groupby(CASE_ID)[TIMESTAMP].agg(["min", "max"])
    spans = spans.sort_values("min")

    previous_end = spans["max"].shift(1)
    assert (spans["min"].iloc[1:] > previous_end.iloc[1:]).all()
