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
