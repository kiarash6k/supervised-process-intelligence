from pathlib import Path

import pandas as pd
import pm4py


def load_event_log(path):
    """
    Load an event log from a CSV or XES file.

    Args:
        path (str or Path): The path to the event log file (.csv, .xes or .xes.gz).

    Returns:
        pd.DataFrame: The loaded event log.

    Raises:
        ValueError: If the file is not a CSV or XES file.
    """
    name = Path(path).name.lower()

    if name.endswith(".csv"):
        event_log = pd.read_csv(path)
    elif name.endswith((".xes", ".xes.gz")):
        event_log = pm4py.read_xes(str(path))
    else:
        raise ValueError(
            f"Unsupported file format for {path}. Please provide a CSV or XES file."
        )

    return event_log
