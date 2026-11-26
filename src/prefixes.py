"""
Prefixes
"""
import pandas as pd

from src.data_loading import CASE_ID, TIMESTAMP
REMAINING_TIME = "remaining_time"
EVENTS = "events" # THERE is no

def extract_prefixes(data: pd.DataFrame, useDay=False)-> pd.DataFrame:
    """
    Extracts prefixes.

    Args:
        data (pd.DataFrame): The input data from which to extract prefixes.

    Returns:
        pd.DataFrame: A DataFrame with case ID, event list, and remaining time.
        Each row contains a prefix (partial sequence of events) for a case.
    """
    prefixes = pd.DataFrame()
    for case_id, group in data.groupby(CASE_ID):
        group = group.sort_values(by=TIMESTAMP)
        for i in range(1, len(group)):
            group_cut = group.iloc[:i]
            processed_group_cut = group_cut.copy()
            processed_group_cut.drop(columns=[CASE_ID], inplace=True)
            group_as_dict = processed_group_cut.to_dict(orient='records')
            # very chuddy implementation
            prefix = {CASE_ID: [case_id], EVENTS: [group_as_dict],
                      REMAINING_TIME: [(group[TIMESTAMP].max() - group_cut[TIMESTAMP].max()).total_seconds() / 3600 if not useDay else (group[TIMESTAMP].max() - group_cut[TIMESTAMP].max()).total_seconds() / 86400]}

            prefixes = pd.concat([prefixes, pd.DataFrame(prefix)], ignore_index=True)
    return prefixes
