"""
Labels

"""
from pandas import DataFrame


def compute_completetion_time(frame: DataFrame):
    """
    Computes the completion time of each case in the event log.

    :param frame: The event log as a pandas DataFrame.
    :return: Should it be a Dataframe? idk
    """
    # Group by case_id and find the maximum timestamp for each case
    completion_times = frame.groupby('case_id')['timestamp'].max()
    return completion_times