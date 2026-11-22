import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_loading import ACTIVITY, CASE_ID, TIMESTAMP, load_config, load_dataset

SECONDS_PER_DAY = 86400


def compute_statistics(event_log: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics of a normalized event log.

    A variant is the complete sequence of activities of a case, so two cases
    have the same variant if they perform the same activities in the same order.
    The duration of a case is the time between its first and its last event.

    Args:
        event_log (pd.DataFrame): A normalized event log (see load_dataset),
            sorted by case and time.

    Returns:
        dict: The number of cases, events, activities and variants, and the mean,
            median and maximum case duration in days.
    """
    cases = event_log.groupby(CASE_ID)
    variants = cases[ACTIVITY].apply(tuple)
    durations = cases[TIMESTAMP].agg(lambda times: times.max() - times.min())
    durations = durations.dt.total_seconds() / SECONDS_PER_DAY

    return {
        "cases": int(event_log[CASE_ID].nunique()),
        "events": int(len(event_log)),
        "activities": int(event_log[ACTIVITY].nunique()),
        "variants": int(variants.nunique()),
        "case_duration_days": {
            "mean": float(durations.mean()),
            "median": float(durations.median()),
            "max": float(durations.max()),
        },
    }


def save_statistics(statistics: dict, path: str | Path) -> None:
    """Write the statistics to a JSON file, creating its folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save descriptive statistics of an event log."
    )
    parser.add_argument("config", help="path to the dataset configuration (YAML)")
    parser.add_argument(
        "--output",
        help="where to save the statistics (default: artifacts/statistics_<name>.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output or f"artifacts/statistics_{config['name']}.json"

    statistics = compute_statistics(load_dataset(args.config))
    save_statistics(statistics, output)
    print(json.dumps(statistics, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
