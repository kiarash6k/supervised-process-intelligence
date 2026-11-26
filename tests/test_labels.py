from data_loading import CASE_ID, TIMESTAMP
from src.labels import append_completion_time_labels


def test_completion_time_appending_columns(synthetic_log):
    """Test that completion time labels are appended with correct columns."""
    res = append_completion_time_labels(synthetic_log)
    expected_columns = {"case:concept:name", "concept:name", "time:timestamp", "time:timestamp_completion", "customer"}
    actual_columns = set(res.columns)
    assert expected_columns.issubset(actual_columns)


def test_completion_time_correct_time(synthetic_log):
    """Test that the completion time is correct for each case."""
    res = append_completion_time_labels(synthetic_log)
    
    for case_id, group in res.groupby(CASE_ID):
        expected_completion_time = group[TIMESTAMP].max()
        actual_completion_times = group["time:timestamp_completion"].unique()
        assert len(actual_completion_times) == 1
        assert actual_completion_times[0] == expected_completion_time


def test_completion_time_is_always_latest(synthetic_log):
    res = append_completion_time_labels(synthetic_log)
    
    for case_id, group in res.groupby(CASE_ID):
        completion_time = group["time:timestamp_completion"].iloc[0]
        max_event_time = group[TIMESTAMP].max()
        assert completion_time == max_event_time