"""
Features
"""

from collections import Counter
from math import cos, pi, sin

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_loading import ACTIVITY, CASE_ID, TIMESTAMP
from src.prefixes import EVENTS, REMAINING_TIME


def extract_prefix_features(events: list[dict],*, static_attributes: list[str], time_unit: str = "hours", reassignment_field: str | None = None) -> dict:
    """
    Extract raw features from one observed prefix in chronological order.

    Args:
        events (list[dict]): Nonempty event records with standardized activity
            names and parsed timestamps, ordered from earliest to latest.
        static_attributes (list[str]): Attributes to copy from the first event.
        time_unit (str): Unit for elapsed durations, either "hours" or "days".
        reassignment_field (str | None): Optional field identifying the assignee.
            A change between consecutive known assignees counts as a reassignment.
            Missing values are skipped.

    Returns:
        dict: Initial static values, prefix length, elapsed time, time since
            the previous event (zero for a single event), last activity,
            activity counts, and cyclic weekday and hour features. Includes
            the reassignment count when an assignment field is configured.

    Raises:
        ValueError: If the prefix is empty, the time unit is unsupported, or
            a static attribute name conflicts with a computed feature.
        KeyError: If a required or configured field is absent.
    """
    if not events:
        raise ValueError("Cannot extract features from an empty prefix.")

    seconds_per_unit = {"hours": 3600, "days": 86400}
    if time_unit not in seconds_per_unit:
        raise ValueError("time_unit must be 'hours' or 'days'.")
    divisor = seconds_per_unit[time_unit]

    first = events[0]
    last = events[-1]
    timestamp = last[TIMESTAMP]
    
    if len(events) > 1:
        previous = events[-2][TIMESTAMP]
    else:
        previous = timestamp

    weekday_angle = 2 * pi * timestamp.weekday() / 7
    hour_angle = 2 * pi * timestamp.hour / 24

    features = {
        "prefix_length": len(events),
        "elapsed_time": (timestamp - first[TIMESTAMP]).total_seconds() / divisor,
        "time_since_previous": (timestamp - previous).total_seconds() / divisor,
        "last_activity": last[ACTIVITY],
        "activity_counts": dict(Counter(event[ACTIVITY] for event in events)),
        "weekday_sin": sin(weekday_angle),
        "weekday_cos": cos(weekday_angle),
        "hour_sin": sin(hour_angle),
        "hour_cos": cos(hour_angle),
    }
    if reassignment_field is not None:
        reassignments = 0
        previous_assignee = None
        for event in events:
            assignee = event[reassignment_field]
            if pd.isna(assignee):
                continue
            if previous_assignee is not None and previous_assignee != assignee:
                reassignments += 1
            previous_assignee = assignee
        features["reassignment_count"] = reassignments

    for attribute in static_attributes:
        if attribute in features:
            raise ValueError(f"Duplicate feature name: {attribute}")
        features[attribute] = first[attribute]
    return features


def build_feature_table(prefixes: pd.DataFrame, *, static_attributes: list[str], time_unit: str = "hours", reassignment_field: str | None = None) -> pd.DataFrame:
    """
    Build one raw feature row per prefix, preserving input order and index.

    Args:
        prefixes (pd.DataFrame): Prefix records with event lists in EVENTS.
        static_attributes (list[str]): Attributes to copy from each first event.
        time_unit (str): Unit for durations, either "hours" or "days".
        reassignment_field (str | None): Optional field identifying the assignee.

    Returns:
        pd.DataFrame: Raw features aligned with the input rows. An empty input
            produces an empty table with the same index and no feature columns.
    """
    if len(prefixes.index) == 0:
        return pd.DataFrame(index=prefixes.index)

    rows = [
        extract_prefix_features(
            events,
            static_attributes=static_attributes,
            time_unit=time_unit,
            reassignment_field=reassignment_field,
        )
        for events in prefixes[EVENTS]
    ]
    return pd.DataFrame(rows, index=prefixes.index)


def _categorical_values(frame: pd.DataFrame) -> np.ndarray:
    """Represent category IDs consistently and mark missing values for imputation."""
    return frame.astype("string").to_numpy(dtype=object, na_value=np.nan)


def build_preprocessor(*, categorical_attributes=(), numeric_attributes=(), include_reassignment: bool = False, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build an unfitted transformer for the output of build_feature_table.

    Args:
        categorical_attributes: Static fields to one-hot encode, including IDs.
        numeric_attributes: Additional numeric static fields to include.
        include_reassignment: Include the count when reassignment_field was set.
        scale_numeric: Standardize numeric features and scale sparse counts.

    Returns:
        ColumnTransformer: Fit on training prefixes and reuse for prediction.
            Unlisted input columns are dropped.
    """
    numeric = [
        "prefix_length", "elapsed_time", "time_since_previous",
        "weekday_sin", "weekday_cos", "hour_sin", "hour_cos",
        *numeric_attributes,
    ]
    if include_reassignment:
        numeric.append("reassignment_count")
    categorical = ["last_activity", *categorical_attributes]
    selected = [*numeric, *categorical, "activity_counts"]
    if len(selected) != len(set(selected)):
        raise ValueError("Each feature must be assigned to exactly one column.")
    if {CASE_ID, REMAINING_TIME, EVENTS}.intersection(selected):
        raise ValueError("Case IDs, targets and event lists cannot be features.")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categorical_pipe = Pipeline([
        ("normalize", FunctionTransformer(
            _categorical_values, feature_names_out="one-to-one"
        )),
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    count_pipe = Pipeline([
        ("vectorizer", DictVectorizer()),
        ("scaler", StandardScaler(with_mean=False) if scale_numeric else "passthrough"),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("category", categorical_pipe, categorical),
        ("count", count_pipe, "activity_counts"),
    ])
