import json

import pandas as pd
import pytest

from src.data_loading import CASE_ID, TIMESTAMP
from src.prefixes import extract_prefixes
from src.split import save_split_report, split_prefixes


def test_chronology_case_integrity_and_saved_report(synthetic_log, tmp_path):
    prefixes = extract_prefixes(synthetic_log)
    parts, report = split_prefixes(synthetic_log, prefixes)
    expected = {"train": ["c1", "c2"], "validation": ["c3"], "test": ["c4"]}
    for name, ids in expected.items():
        assert report[name]["case_ids"] == ids
        rows = prefixes.loc[prefixes[CASE_ID].isin(ids)]
        pd.testing.assert_frame_equal(parts[name], rows)
        assert report[name]["case_count"] == len(ids)
        assert report[name]["prefix_count"] == len(rows)
        starts = synthetic_log[synthetic_log[CASE_ID].isin(ids)].groupby(CASE_ID)[TIMESTAMP].min()
        assert report[name]["start_min"] == starts.min().isoformat()
        assert report[name]["start_max"] == starts.max().isoformat()
    path = tmp_path / "artifacts" / "split.json"
    save_split_report(report, path)
    assert json.loads(path.read_text()) == report


def test_default_and_custom_shares_with_equal_starts():
    log = pd.DataFrame({CASE_ID: range(20), TIMESTAMP: pd.Timestamp("2026-01-01")})
    empty = extract_prefixes(log)
    _, default = split_prefixes(log.sample(frac=1, random_state=7), empty)
    assert [p["case_count"] for p in default.values()] == [14, 3, 3]
    assert default["train"]["case_ids"] == list(range(14))
    assert all(p["prefix_count"] == 0 for p in default.values())
    _, custom = split_prefixes(log, empty, train=.5, validation=.25, test=.25)
    assert [p["case_count"] for p in custom.values()] == [10, 5, 5]


def test_overlapping_case_is_assigned_by_first_event(synthetic_log):
    log = synthetic_log.copy()
    first = log[CASE_ID] == "c1"
    log.loc[log[first][TIMESTAMP].idxmax(), TIMESTAMP] = pd.Timestamp("2021-01-01")
    parts, report = split_prefixes(log, extract_prefixes(log))
    assert report["train"]["case_ids"][0] == "c1"
    assert (parts["train"][CASE_ID] == "c1").sum() == 2


@pytest.mark.parametrize("shares", [(0, .5, .5), (.8, .2, .2), (float("nan"), .5, .5)])
def test_invalid_shares(synthetic_log, shares):
    with pytest.raises(ValueError, match="shares"):
        split_prefixes(synthetic_log, pd.DataFrame(), train=shares[0], validation=shares[1], test=shares[2])


def test_too_few_cases(synthetic_log):
    log = synthetic_log[synthetic_log[CASE_ID].isin(["c1", "c2"])]
    with pytest.raises(ValueError, match="three cases"):
        split_prefixes(log, extract_prefixes(log))


def test_unknown_prefix_case(synthetic_log):
    with pytest.raises(ValueError, match="absent"):
        split_prefixes(synthetic_log, pd.DataFrame({CASE_ID: ["unknown"]}))
