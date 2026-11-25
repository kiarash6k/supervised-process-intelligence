from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
import yaml

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"

class EventLogError(ValueError):
    """The event log is empty, has a missing column, or has an invalid timestamp."""


def load_config(path: str | Path) -> dict:
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


def load_event_log(path: str | Path) -> pd.DataFrame:
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


def standardize_columns(event_log: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Rename the case, activity and timestamp columns to the standard names.

    The names of the three columns are read from the dataset configuration:

        columns:
          case_id: "Case ID"
          activity: "Activity"
          timestamp: "Complete Timestamp"
          resource : "Resource"
    They are renamed to CASE_ID, ACTIVITY and TIMESTAMP. All other columns are
    kept unchanged. A column named in the configuration that does not exist in
    the event log is not reported here; call validate_event_log first.

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
        columns["resource"]: RESOURCE,
    }

    return event_log.rename(columns=column_mapping)


def validate_event_log(event_log: pd.DataFrame, config: dict) -> None:
    """
    Check a raw event log before its columns are renamed.

    Args:
        event_log (pd.DataFrame): The event log as loaded from the file.
        config (dict): The dataset configuration with a "columns" section.

    Raises:
        EventLogError: If the log has no events, or if one of the case, activity,
            timestamp or static attribute columns named in the configuration is
            missing.
    """
    if event_log.empty:
        raise EventLogError("The event log is empty.")
    required = [
        config["columns"]["case_id"],
        config["columns"]["activity"],
        config["columns"]["timestamp"],
        config["columns"]["resource"],
    ]
    required += list(config.get("static_attributes") or [])
    missing = [name for name in required if name not in event_log.columns]
    if missing:
        raise EventLogError(
            f"Required column(s) missing in the event log: {', '.join(missing)}."
        )


def parse_timestamps(event_log: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Convert the timestamp column to datetimes.

    The format is read from config["timestamp_format"] (a strptime pattern such
    as "%Y/%m/%d %H:%M:%S.%f"). If it is missing, pandas infers the format. A
    column that already contains datetimes (for example from an XES file) is
    returned unchanged.

    Args:
        event_log (pd.DataFrame): An event log with the standard column names.
        config (dict): The dataset configuration.

    Returns:
        pd.DataFrame: A new event log with a datetime TIMESTAMP column.

    Raises:
        EventLogError: If a timestamp is missing or does not match the format.
    """
    event_log = event_log.copy()
    if pd.api.types.is_datetime64_any_dtype(event_log[TIMESTAMP]):
        return event_log

    parsed = pd.to_datetime(
        event_log[TIMESTAMP], format=config.get("timestamp_format"), errors="coerce"
    )
    failed = parsed.isna()
    if failed.any():
        first = event_log.loc[failed, TIMESTAMP].iloc[0]
        raise EventLogError(
            f"{int(failed.sum())} timestamp(s) cannot be parsed with the format "
            f"{config.get('timestamp_format')!r}; the first one is {first!r}."
        )

    event_log[TIMESTAMP] = parsed
    return event_log


def sort_events(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the events by case and then by timestamp.

    Events of the same case with the same timestamp keep the order they have in
    the input (the file order).

    Args:
        event_log (pd.DataFrame): An event log with parsed timestamps.

    Returns:
        pd.DataFrame: A new event log with a fresh index 0..n-1.
    """
    position = "_position"
    event_log = event_log.assign(**{position: np.arange(len(event_log))})
    event_log = event_log.sort_values([CASE_ID, TIMESTAMP, position])
    return event_log.drop(columns=position).reset_index(drop=True)


def load_dataset(config_path: str | Path) -> pd.DataFrame:
    """
    Load an event log through its dataset configuration and normalize it.

    The steps are: read the configuration, load the file, validate it, rename
    the case, activity and timestamp columns to the standard names, parse the
    timestamps, and sort the events by case and time. Only the three standard
    columns and the static attributes named in the configuration are kept, so
    columns such as "Variant" (which describe the whole case, including its
    future) never reach the features.

    Args:
        config_path (str or Path): The path to the dataset configuration (YAML).

    Returns:
        pd.DataFrame: The normalized event log with the columns CASE_ID,
            ACTIVITY, TIMESTAMP and the static attributes.

    Raises:
        EventLogError: If the log is empty, a configured column is missing, or
            a timestamp cannot be parsed.
        ValueError: If the configuration or the log has an unsupported format.
    """
    config = load_config(config_path)

    event_log = load_event_log(config["path"])
    validate_event_log(event_log, config)

    event_log = standardize_columns(event_log, config)
    event_log = parse_timestamps(event_log, config)
    event_log = sort_events(event_log)

    static_attributes = list(config.get("static_attributes") or [])
    return event_log[[CASE_ID, ACTIVITY, TIMESTAMP, RESOURCE, *static_attributes]]