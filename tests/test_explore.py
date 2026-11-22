import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import ACTIVITY, CASE_ID, TIMESTAMP, load_dataset
from src.explore import compute_statistics, save_statistics


def make_log(rows):
    log = pd.DataFrame(rows, columns=[CASE_ID, ACTIVITY, TIMESTAMP])
    log[TIMESTAMP] = pd.to_datetime(log[TIMESTAMP])
    return log


@pytest.fixture
def small_log():
    # c1: A,B over 2 days; c2: A,B over 4 days (same variant as c1);
    # c3: A only (duration 0)
    return make_log(
        [
            ("c1", "A", "2020-01-01 00:00:00"),
            ("c1", "B", "2020-01-03 00:00:00"),
            ("c2", "A", "2020-01-01 00:00:00"),
            ("c2", "B", "2020-01-05 00:00:00"),
            ("c3", "A", "2020-01-02 00:00:00"),
        ]
    )


def test_compute_statistics_counts(small_log):
    stats = compute_statistics(small_log)

    assert stats["cases"] == 3
    assert stats["events"] == 5
    assert stats["activities"] == 2
    assert stats["variants"] == 2  # (A, B) and (A,)


def test_compute_statistics_case_durations(small_log):
    durations = compute_statistics(small_log)["case_duration_days"]

    assert durations["mean"] == pytest.approx(2.0)  # (2 + 4 + 0) / 3
    assert durations["median"] == pytest.approx(2.0)
    assert durations["max"] == pytest.approx(4.0)


def test_different_order_is_a_different_variant():
    log = make_log(
        [
            ("c1", "A", "2020-01-01"),
            ("c1", "B", "2020-01-02"),
            ("c2", "B", "2020-01-01"),
            ("c2", "A", "2020-01-02"),
        ]
    )

    assert compute_statistics(log)["variants"] == 2


def test_save_statistics_writes_json(tmp_path, small_log):
    stats = compute_statistics(small_log)
    path = tmp_path / "out" / "stats.json"

    save_statistics(stats, path)

    assert json.loads(path.read_text(encoding="utf-8")) == stats


HELPDESK_LOG = Path("data/help-desk-log.csv")


@pytest.mark.skipif(not HELPDESK_LOG.exists(), reason="Helpdesk log is not downloaded")
def test_helpdesk_statistics():
    stats = compute_statistics(load_dataset("config/helpdesk.yaml"))

    assert stats["cases"] == 4580
    assert stats["events"] == 21348
    assert stats["activities"] == 14
    assert stats["variants"] > 1
    durations = stats["case_duration_days"]
    assert 0 <= durations["median"] <= durations["max"]
