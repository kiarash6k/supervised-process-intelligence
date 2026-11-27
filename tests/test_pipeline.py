import numpy as np
import pandas as pd
import pytest

from src.pipeline import compute_errors, run_pipeline

SYNTHETIC_CONFIG = "tests/data/synthetic.yaml"


def test_compute_errors():
    errors = compute_errors(np.array([10.0, 20.0]), np.array([13.0, 16.0]))

    assert errors["mae_hours"] == pytest.approx(3.5)  # (3 + 4) / 2
    assert errors["rmse_hours"] == pytest.approx(np.sqrt(12.5))  # (9 + 16) / 2


def test_pipeline_runs_from_the_raw_log_to_the_results(tmp_path):
    results = run_pipeline(SYNTHETIC_CONFIG, output_dir=tmp_path)

    assert set(zip(results["model"], results["set"])) == {
        ("mean_baseline", "validation"),
        ("mean_baseline", "test"),
        ("ridge", "validation"),
        ("ridge", "test"),
    }
    assert np.isfinite(results[["mae_hours", "rmse_hours"]].to_numpy()).all()
    for file_name in (
        "split_synthetic.json",
        "predictions_synthetic.csv",
        "results_synthetic.csv",
    ):
        assert (tmp_path / file_name).exists()


def test_pipeline_predictions(tmp_path):
    run_pipeline(SYNTHETIC_CONFIG, output_dir=tmp_path)
    predictions = pd.read_csv(tmp_path / "predictions_synthetic.csv")

    # the training targets are c1 (72, 48) and c2 (48, 24), so their mean is 48
    baseline = predictions[predictions["model"] == "mean_baseline"]
    np.testing.assert_allclose(baseline["predicted_hours"], 48.0)
    assert (predictions["predicted_hours"] >= 0).all()
    # training cases are never used to measure the models
    assert set(predictions["case_id"]) == {"c3", "c4"}


def test_pipeline_gives_the_same_results_twice(tmp_path):
    first = run_pipeline(SYNTHETIC_CONFIG, output_dir=tmp_path / "first")
    second = run_pipeline(SYNTHETIC_CONFIG, output_dir=tmp_path / "second")

    pd.testing.assert_frame_equal(first, second)
