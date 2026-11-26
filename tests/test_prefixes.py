from src.prefixes import extract_prefixes

def test_extract_prefixes_returns_dataframe(synthetic_log):
    """Test that extract_prefixes returns a DataFrame with the correct structure."""
    prefixes = extract_prefixes(synthetic_log)
    
    assert isinstance(prefixes, type(synthetic_log))
    assert "case:concept:name" in prefixes.columns
    assert "events" in prefixes.columns
    assert "remaining_time" in prefixes.columns
    assert len(prefixes) > 0


def test_extract_prefixes_events_are_lists(synthetic_log):
    """Test that each event column contains a list of event dictionaries."""
    prefixes = extract_prefixes(synthetic_log)
    
    for idx in range(len(prefixes)):
        events = prefixes.iloc[idx]["events"]
        assert isinstance(events, list), f"Row {idx}: events should be a list, got {type(events)}"
        assert len(events) > 0, f"Row {idx}: events list should not be empty"
        for event in events:
            assert isinstance(event, dict), f"Row {idx}: each event should be a dict"


def test_extract_prefixes_remaining_time_is_numeric(synthetic_log):
    """Test that remaining_time is numeric (float or int)."""
    prefixes = extract_prefixes(synthetic_log)
    
    for idx in range(len(prefixes)):
        remaining_time = prefixes.iloc[idx]["remaining_time"]
        assert isinstance(remaining_time, (int, float)), f"Row {idx}: remaining_time should be numeric"
        assert remaining_time >= 0, f"Row {idx}: remaining_time should be non-negative"


def test_extract_prefixes_number_of_rows(synthetic_log):
    """Test that the number of prefix rows matches expectations."""
    prefixes = extract_prefixes(synthetic_log)
    
    # For synthetic_log: 4 cases with 3, 3, 4, 3 events respectively
    # Number of prefixes = (3-1) + (3-1) + (4-1) + (3-1) = 2 + 2 + 3 + 2 = 9
    expected_rows = sum(len(group) - 1 for _, group in synthetic_log.groupby("case:concept:name"))
    assert len(prefixes) == expected_rows