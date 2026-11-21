import pandas as pd
import pm4py
import pytest

from src.data_loading import load_event_log


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
