import numpy as np
import pandas as pd

from src.data_loading import CASE_ID
from src.models import build_mean_predictor
from src.prefixes import REMAINING_TIME, extract_prefixes
from src.split import split_prefixes


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