from pathlib import Path

import pandas as pd
import pm4py
import yaml

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"


def load_config(path):
    """
    Load a YAML dataset configuration and return its content as a dictionary.

    Args:
        path (str or Path): The path to the YAML file (.yaml or .yml).

    Returns:
        dict: The content of the YAML file.

    Raises:
        ValueError: If the file is not a Yaml file.
    """
    name = Path(path).name.lower()

    if not name.endswith((".yaml", ".yml")):
        raise ValueError(
            f"Unsupported file format for {path}. Please provide a Yaml file."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


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


def standardize_columns(event_log, config):
    """
    Rename the case, activity and timestamp columns to the standard names.

    The names of the three columns are read from the dataset configuration:

        columns:
          case_id: "Case ID"
          activity: "Activity"
          timestamp: "Complete Timestamp"

    They are renamed to CASE_ID, ACTIVITY and TIMESTAMP. All other columns are
    kept unchanged. A column named in the configuration that does not exist in
    the event log is not reported here (the validation does that).

    Args:
        event_log (pd.DataFrame): The event log to standardize.
        config (dict): The dataset configuration with a "columns" section.

    Returns:
        pd.DataFrame: A new event log with the standard column names.

    Raises:
        KeyError: If the configuration has no "columns" section or one of its
            three keys is missing.
    """
    columns = config["columns"]
    column_mapping = {
        columns["case_id"]: CASE_ID,
        columns["activity"]: ACTIVITY,
        columns["timestamp"]: TIMESTAMP,
    }

    return event_log.rename(columns=column_mapping)