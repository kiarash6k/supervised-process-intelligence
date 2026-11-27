"""Chronological train, validation and test sets for case prefixes."""

import json
from math import floor, isclose, isfinite
from pathlib import Path

import pandas as pd

from src.data_loading import CASE_ID, TIMESTAMP


def split_prefixes(event_log: pd.DataFrame, prefixes: pd.DataFrame, *, train: float = 0.70, validation: float = 0.15, test: float = 0.15) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Assign whole cases by start time, then select their existing prefix rows.

    Args:
        event_log: Normalized log with a case identifier and timestamp per event.
        prefixes: Output of extract_prefixes for this log.
        train, validation, test: Positive shares of cases summing to one.

    Returns:
        Prefix tables and a JSON-compatible report keyed by partition name.
        Each table retains the input prefix order and index. Equal starts are
        ordered by case ID. Each partition receives at least one case.

    Raises:
        ValueError: For invalid shares, missing identifiers or times, fewer
            than three cases, or prefix identifiers absent from the log.
    """
    shares = (train, validation, test)
    if not all(isfinite(s) and s > 0 for s in shares) or not isclose(sum(shares), 1):
        raise ValueError("Split shares must be positive and sum to one.")
    if event_log[[CASE_ID, TIMESTAMP]].isna().any().any():
        raise ValueError("Case identifiers and timestamps must be present.")

    starts = event_log.groupby(CASE_ID, as_index=False, sort=False)[TIMESTAMP].min()
    starts = starts.sort_values([TIMESTAMP, CASE_ID])
    n = len(starts)
    if n < 3:
        raise ValueError("At least three cases are required.")
    
    # extract_prefixes returns a columnless table if all cases have one event.
    if prefixes.empty and CASE_ID not in prefixes:
        prefixes = prefixes.assign(**{CASE_ID: pd.Series(dtype=event_log[CASE_ID].dtype)})
    if not prefixes[CASE_ID].isin(starts[CASE_ID]).all():
        raise ValueError("A prefix refers to a case absent from the event log.")

    # Round by largest deficit while reserving one case for each partition.
    quotas = [n * share for share in shares] # 1. Calculate the number of prefixes in each group
    sizes = [max(1, floor(quota)) for quota in quotas] # 2. Rounding the values 

    # 3. If the sizes are too big, decrease them
    while sum(sizes) > n:
        # Choosing the biggest group as a donor
        donor = sizes.index(max(sizes))
        sizes[donor] -= 1
    # Give it to the group with the highest deficit 
    while sum(sizes) < n:
        receiver = max(range(3), key=lambda i: quotas[i] - sizes[i])
        sizes[receiver] += 1

    partitions = {}
    report = {}
    offset = 0
    for name, size in zip(("train", "validation", "test"), sizes):
        cases = starts.iloc[offset:offset + size]
        offset += size
        ids = cases[CASE_ID].tolist()
        rows = prefixes.loc[prefixes[CASE_ID].isin(ids)].copy()
        partitions[name] = rows
        report[name] = {
            "case_ids": ids,
            "start_min": cases[TIMESTAMP].iloc[0].isoformat(),
            "start_max": cases[TIMESTAMP].iloc[-1].isoformat(),
            "case_count": size,
            "prefix_count": len(rows),
        }
    return partitions, report


def save_split_report(report: dict, path: str | Path) -> None:
    """Save case membership, inclusive start boundaries and counts as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
