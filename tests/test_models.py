import numpy as np
import pandas as pd
import pytest

from src.data_loading import CASE_ID
from src.models import build_mean_predictor, build_ridge_regressor
from src.prefixes import REMAINING_TIME, extract_prefixes
from src.split import split_prefixes
from src.features import build_feature_table
from src.data_loading import load_config


def test_mean_predictor_weights_each_prefix_equally():
    # Case A contributes two prefixes and case B contributes one.
    train = pd.DataFrame({CASE_ID: ["A", "A", "B"], REMAINING_TIME: [30, 21, 3]})
    model = build_mean_predictor()
    model.fit(train[[CASE_ID]], train[REMAINING_TIME])

    new = pd.DataFrame({CASE_ID: ["C", "D"]})
    np.testing.assert_allclose(model.predict(new), [18, 18])


def test_mean_predictor_uses_only_training_targets(synthetic_log):
    prefixes = extract_prefixes(synthetic_log)
    parts, _ = split_prefixes(synthetic_log, prefixes)
    train = parts["train"]
    model = build_mean_predictor()
    model.fit(train[[CASE_ID]], train[REMAINING_TIME])

    # Training targets in hours: c1 (72, 48), c2 (48, 24).
    for name in ("validation", "test"):
        held_out = parts[name]
        predictions = model.predict(held_out[[CASE_ID]])
        np.testing.assert_allclose(predictions, np.full(len(held_out), 48.0))

@pytest.mark.parametrize("alpha", [1.0, 6.0])
def test_ridge_regularizes_a_single_numeric_feature(synthetic_log, alpha):
    prefixes = extract_prefixes(synthetic_log)
    row = build_feature_table(prefixes.iloc[:1], static_attributes=[])
    train = pd.concat([row, row], ignore_index=True).assign(cost=[10, 30])
    model = build_ridge_regressor(alpha=alpha, numeric_attributes=["cost"])
    model.fit(train, [2, 6])

    # Scaling maps cost to -1 and +1. All other features are constant.
    # The fitted intercept is 4 and the Ridge slope is 4 / (2 + alpha).
    slope = 4 / (2 + alpha)
    np.testing.assert_allclose(model.predict(train), [4 - slope, 4 + slope])


def test_ridge_predicts_on_held_out_prefixes(synthetic_log):
    parts, _ = split_prefixes(synthetic_log, extract_prefixes(synthetic_log))
    alpha = load_config("config/pipeline.yaml")["ridge"]["alpha"]
    model = build_ridge_regressor(alpha=alpha, categorical_attributes=["customer"])
    train = build_feature_table(parts["train"], static_attributes=["customer"])
    model.fit(train, parts["train"][REMAINING_TIME])
    names = model.named_steps["preprocess"].get_feature_names_out().copy()

    for name in ("validation", "test"):
        raw = build_feature_table(parts[name], static_attributes=["customer"])
        predictions = model.predict(raw)
        assert predictions.shape == (len(raw),)
        assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(model.named_steps["preprocess"].get_feature_names_out(), names)
