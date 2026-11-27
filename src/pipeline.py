"""
Run the basic pipeline: from the raw event log to the errors of the models.

    python -m src.pipeline config/helpdesk.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import CASE_ID, load_config, load_dataset
from src.features import build_feature_table
from src.models import build_mean_predictor, build_ridge_regressor
from src.prefixes import PREFIX_LENGTH, REMAINING_TIME, extract_prefixes
from src.seed import DEFAULT_CONFIG, load_seed, set_seed
from src.split import save_split_report, split_prefixes

# the models are fitted on "train" only; these sets are used to measure them
EVALUATION_SETS = ("validation", "test")


def compute_errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Measure how far the predictions are from the true remaining times.

    Args:
        actual (np.ndarray): The true remaining times.
        predicted (np.ndarray): The predicted remaining times, same order.

    Returns:
        dict: The mean absolute error (MAE) and the root mean squared error
            (RMSE), both in the unit of the input.
    """
    errors = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae_hours": float(np.mean(np.abs(errors))),
        "rmse_hours": float(np.sqrt(np.mean(errors**2))),
    }


def run_pipeline(
    dataset_config: str | Path,
    settings_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = "artifacts",
) -> pd.DataFrame:
    """
    Train the baseline and the Ridge model and evaluate them.

    The steps are: load the log, build the prefixes with their remaining time in
    hours, split the cases by start time, build the features, fit both models on
    the training prefixes, and predict the validation and test prefixes.
    A negative prediction is replaced by 0, because a remaining time cannot be
    negative.

    Three files are written to the output folder, where <name> is the name of
    the dataset: split_<name>.json (which case is in which set),
    predictions_<name>.csv (one row per model and prefix) and
    results_<name>.csv (the errors per model and set).

    Args:
        dataset_config (str or Path): The configuration of the dataset (YAML).
        settings_path (str or Path): The settings of the pipeline (YAML).
        output_dir (str or Path): The folder for the output files.

    Returns:
        pd.DataFrame: The errors, one row per model and set.
    """
    dataset = load_config(dataset_config)
    settings = load_config(settings_path)
    set_seed(load_seed(settings_path))

    name = dataset["name"]
    static_attributes = list(dataset.get("static_attributes") or [])
    reassignment_field = settings["features"].get("reassignment_field")
    output_dir = Path(output_dir)

    event_log = load_dataset(dataset_config)
    prefixes = extract_prefixes(event_log)
    parts, split_report = split_prefixes(event_log, prefixes, **settings["split"])
    save_split_report(split_report, output_dir / f"split_{name}.json")

    features = {
        set_name: build_feature_table(
            part,
            static_attributes=static_attributes,
            time_unit="hours",
            reassignment_field=reassignment_field,
        )
        for set_name, part in parts.items()
    }

    models = {
        "mean_baseline": build_mean_predictor(),
        "ridge": build_ridge_regressor(
            alpha=settings["ridge"]["alpha"],
            categorical_attributes=static_attributes,
            include_reassignment=reassignment_field is not None,
        ),
    }

    results = []
    predictions = []
    for model_name, model in models.items():
        model.fit(features["train"], parts["train"][REMAINING_TIME])
        for set_name in EVALUATION_SETS:
            actual = parts[set_name][REMAINING_TIME].to_numpy(dtype=float)
            predicted = np.clip(model.predict(features[set_name]), 0, None)
            results.append(
                {
                    "model": model_name,
                    "set": set_name,
                    "prefixes": len(actual),
                    **compute_errors(actual, predicted),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "set": set_name,
                        "case_id": parts[set_name][CASE_ID].to_numpy(),
                        "prefix_length": parts[set_name][PREFIX_LENGTH].to_numpy(),
                        "last_activity": features[set_name]["last_activity"].to_numpy(),
                        "actual_hours": actual,
                        "predicted_hours": predicted,
                    }
                )
            )

    results = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / f"results_{name}.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / f"predictions_{name}.csv", index=False
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the remaining time models on an event log."
    )
    parser.add_argument("config", help="path to the dataset configuration (YAML)")
    parser.add_argument(
        "--settings", default=DEFAULT_CONFIG, help="path to the pipeline settings"
    )
    parser.add_argument(
        "--output", default="artifacts", help="folder for the output files"
    )
    args = parser.parse_args()

    results = run_pipeline(args.config, args.settings, args.output)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()
