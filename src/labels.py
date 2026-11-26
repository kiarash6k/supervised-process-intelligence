"""
Labels

"""
from typing import cast

import pandas as pd

from data_loading import load_dataset, CASE_ID, TIMESTAMP


def append_completion_time_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Appends the completion time of each case in the event log.
    (SUPPOSED TO FULFIL REQ-F08)
    :param data: The event log as a pandas DataFrame.
    :return: the same dataframe but with completion timestamp added to it
    """

    completion_times = data.groupby(CASE_ID)[TIMESTAMP].max()
    data_with_completion = pd.merge(data, completion_times, on=CASE_ID, suffixes=('', '_completion'))
    return data_with_completion

def append_remaining_time_to_prefixes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Appends remaining time to each prefix in the event log.
    PREFIXES NEED TO BE EXTRACTED BEFORE THIS.
    :param data: The prefixes as a pandas DataFrame.
    :return: the same dataframe but with remaining time timestamp added to them
    """

    completion_times = data.groupby(CASE_ID)[TIMESTAMP].max()
    data_with_completion = pd.merge(data, completion_times, on=CASE_ID, suffixes=('', '_completion'))
    return data_with_completion