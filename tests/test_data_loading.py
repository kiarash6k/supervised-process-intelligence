from pathlib import Path

import pandas as pd
import pm4py
import pytest

from src.data_loading import (
    ACTIVITY,
    CASE_ID,
    TIMESTAMP,
    EventLogError,
    load_config,
    load_dataset,
    load_event_log,
    parse_timestamps,
    sort_events,
    standardize_columns,
    validate_event_log,
)

CONFIG = {
    "columns": {"case_id": "case", "activity": "act", "timestamp": "time"},
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
}


def make_log(rows):
    """Build a standardized event log from (case, activity, timestamp) rows."""
    return pd.DataFrame(rows, columns=[CASE_ID, ACTIVITY, TIMESTAMP])


def test_load_event_log_csv(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text(
        "case,act,time\n"
        "c1,A,2020-01-01 10:00:00\n"
        "c1,B,2020-01-02 10:00:00\n"
    )

    event_log = load_event_log(path)

    assert isinstance(event_log, pd.DataFrame)
    assert len(event_log) == 2
    assert list(event_log.columns) == ["case", "act", "time"]


def test_load_event_log_xes(tmp_path):
    events = pd.DataFrame(
        {
            "case:concept:name": ["c1", "c1", "c2"],
            "concept:name": ["A", "B", "A"],
            "time:timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"], utc=True
            ),
        }
    )
    path = tmp_path / "log.xes"
    pm4py.write_xes(events, str(path))

    event_log = load_event_log(path)

    assert isinstance(event_log, pd.DataFrame)
    assert len(event_log) == 3
    assert set(event_log["concept:name"]) == {"A", "B"}


def test_load_event_log_invalid_format(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("not an event log")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_event_log(path)


def test_standardize_columns_with_extra_column():
    # Create a sample DataFrame with extra column
    data = {
        "case": ["c1", "c2"],
        "act": ["A", "B"],
        "time": ["2020-01-01 10:00:00", "2020-01-02 10:00:00"],
        "customer": ["cust1", "cust2"],
    }
    event_log = pd.DataFrame(data)

    # Configuration for standardization
    config = {
        "columns": {
            "case_id": "case",
            "activity": "act",
            "timestamp": "time",
        }
    }

    standardized_log = standardize_columns(event_log, config)

    # Check that the standardized columns are correct
    assert list(standardized_log.columns) == [CASE_ID, ACTIVITY, TIMESTAMP, "customer"]
    assert standardized_log[CASE_ID].tolist() == ["c1", "c2"]
    assert standardized_log[ACTIVITY].tolist() == ["A", "B"]
    assert standardized_log["customer"].tolist() == ["cust1", "cust2"]

    # Check that the original DataFrame is unchanged
    assert list(event_log.columns) == ["case", "act", "time", "customer"]


def test_standardize_columns_with_helpdesk_names():
    event_log = pd.DataFrame(
        {
            "Case ID": ["Case 1"],
            "Activity": ["Assign seriousness"],
            "Complete Timestamp": ["2012/01/01 10:00:00.000"],
        }
    )
    config = {
        "columns": {
            "case_id": "Case ID",
            "activity": "Activity",
            "timestamp": "Complete Timestamp",
        }
    }

    standardized_log = standardize_columns(event_log, config)

    assert list(standardized_log.columns) == [CASE_ID, ACTIVITY, TIMESTAMP]
    assert standardized_log[ACTIVITY].tolist() == ["Assign seriousness"]


def test_load_config(tmp_path):
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "name: demo\n"
        "columns:\n"
        '  case_id: "Case ID"\n'
        '  activity: "Activity"\n'
        '  timestamp: "Complete Timestamp"\n'
        "static_attributes: [customer, product]\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["name"] == "demo"
    assert config["columns"]["case_id"] == "Case ID"
    assert config["static_attributes"] == ["customer", "product"]


def test_load_config_invalid_format(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("name: demo")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_config(path)


def test_parse_timestamps_returns_datetimes():
    event_log = make_log([("c1", "A", "2020-01-01 10:00:00")])

    parsed = parse_timestamps(event_log, CONFIG)

    assert pd.api.types.is_datetime64_any_dtype(parsed[TIMESTAMP])
    assert parsed[TIMESTAMP].iloc[0] == pd.Timestamp("2020-01-01 10:00:00")
    # the input is not modified
    assert event_log[TIMESTAMP].iloc[0] == "2020-01-01 10:00:00"


def test_parse_timestamps_uses_the_configured_format():
    event_log = make_log([("c1", "A", "01/02/2020 10:00")])
    config = {"timestamp_format": "%d/%m/%Y %H:%M"}

    parsed = parse_timestamps(event_log, config)

    assert parsed[TIMESTAMP].iloc[0] == pd.Timestamp("2020-02-01 10:00")


def test_sort_events_orders_events_within_each_case_by_time():
    event_log = parse_timestamps(
        make_log(
            [
                ("c1", "C", "2020-01-03 10:00:00"),
                ("c2", "A", "2020-01-01 09:00:00"),
                ("c1", "A", "2020-01-01 10:00:00"),
                ("c1", "B", "2020-01-02 10:00:00"),
            ]
        ),
        CONFIG,
    )

    sorted_log = sort_events(event_log)

    c1 = sorted_log[sorted_log[CASE_ID] == "c1"]
    assert c1[ACTIVITY].tolist() == ["A", "B", "C"]
    assert sorted_log.index.tolist() == [0, 1, 2, 3]


def test_sort_events_keeps_the_file_order_for_tied_timestamps():
    event_log = parse_timestamps(
        make_log(
            [
                ("c1", "Y", "2020-01-02 10:00:00"),
                ("c1", "X", "2020-01-02 10:00:00"),  # same time, X after Y in the file
                ("c1", "A", "2020-01-01 10:00:00"),
            ]
        ),
        CONFIG,
    )

    sorted_log = sort_events(event_log)

    assert sorted_log[ACTIVITY].tolist() == ["A", "Y", "X"]


def test_validate_event_log_accepts_a_valid_log():
    event_log = pd.DataFrame(
        {"case": ["c1"], "act": ["A"], "time": ["2020-01-01 10:00:00"]}
    )

    validate_event_log(event_log, CONFIG)  # must not raise


def test_validate_event_log_rejects_an_empty_log():
    event_log = pd.DataFrame(columns=["case", "act", "time"])

    with pytest.raises(EventLogError, match="empty"):
        validate_event_log(event_log, CONFIG)


def test_validate_event_log_rejects_a_missing_column():
    event_log = pd.DataFrame({"case": ["c1"], "time": ["2020-01-01 10:00:00"]})

    with pytest.raises(EventLogError, match="act"):
        validate_event_log(event_log, CONFIG)


def test_parse_timestamps_rejects_an_unparsable_timestamp():
    event_log = make_log(
        [("c1", "A", "2020-01-01 10:00:00"), ("c1", "B", "not a date")]
    )

    with pytest.raises(EventLogError, match="not a date"):
        parse_timestamps(event_log, CONFIG)


def write_dataset(tmp_path, static_attributes="[customer]"):
    """Write a small CSV log and its configuration; return the config path."""
    log_path = tmp_path / "log.csv"
    log_path.write_text(
        "case,act,time,customer,Variant\n"
        "c2,A,2020-01-01 09:00:00,bob,V1\n"
        "c1,B,2020-01-02 10:00:00,ann,V2\n"
        "c1,A,2020-01-01 10:00:00,ann,V2\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        f"path: {log_path.as_posix()}\n"
        "columns:\n"
        "  case_id: case\n"
        "  activity: act\n"
        "  timestamp: time\n"
        'timestamp_format: "%Y-%m-%d %H:%M:%S"\n'
        f"static_attributes: {static_attributes}\n",
        encoding="utf-8",
    )
    return config_path


def test_load_dataset_normalizes_the_log(tmp_path):
    log = load_dataset(write_dataset(tmp_path))

    # standard names, static attributes kept, "Variant" dropped
    assert list(log.columns) == [CASE_ID, ACTIVITY, TIMESTAMP, "customer"]
    # sorted by case and time
    assert log[CASE_ID].tolist() == ["c1", "c1", "c2"]
    assert log[ACTIVITY].tolist() == ["A", "B", "A"]
    assert pd.api.types.is_datetime64_any_dtype(log[TIMESTAMP])


def test_load_dataset_rejects_a_missing_static_attribute(tmp_path):
    config_path = write_dataset(tmp_path, static_attributes="[customer, product]")

    with pytest.raises(EventLogError, match="product"):
        load_dataset(config_path)


HELPDESK_LOG = Path("data/help-desk-log.csv")


def test_helpdesk_log_can_be_loaded_and_normalized():
    log = load_dataset("config/helpdesk.yaml")
    assert log[CASE_ID].nunique() == 4580
    assert len(log) == 21348
    assert log[ACTIVITY].nunique() == 14
    assert list(log.columns) == [
        CASE_ID,
        ACTIVITY,
        TIMESTAMP,
        "customer",
        "product",
        "responsible_section",
        "support_section",
    ]
    assert pd.api.types.is_datetime64_any_dtype(log[TIMESTAMP])
    # events are in time order within each case
    assert log.groupby(CASE_ID)[TIMESTAMP].apply(
        lambda times: times.is_monotonic_increasing
    ).all()

