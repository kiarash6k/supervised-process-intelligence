"""
Labels

"""
import pandas as pd

from data_loading import load_dataset


def compute_completetion_time(frame: pd.DataFrame):
    """
    Computes the completion time of each case in the event log.

    :param frame: The event log as a pandas DataFrame.
    :return: Should it be a Dataframe? idk
    """
    # Group by case_id and find the maximum timestamp for each case
    completion_times = frame.groupby('case_id')['timestamp'].max()
    return completion_times
def extract_labels(data: pd.DataFrame) -> list:
    """
    """
    labels = data.groupby('case_id')['label'].unique()
    # Logic to extract labels goes here
    return labels

if __name__ == "__main__":
    # Example usage    data = pd.DataFrame({
    # do example with the proper relative path to
    log = load_dataset("config/helpdesk.yaml")
    labels = extract_labels(log)
    print("Extracted Labels:")
    print(labels)
