import pandas as pd
import pm4py
import pytest

from src.data_loading import (
    ACTIVITY,
    CASE_ID,
    TIMESTAMP,
    load_config,
    load_event_log,
    standardize_columns,
)


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

