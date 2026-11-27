"""Models for predicting the remaining time of case prefixes."""

from sklearn.dummy import DummyRegressor


def build_mean_predictor() -> DummyRegressor:
    """
    Build the mean remaining-time baseline (REQ-F25).

    Returns:
        DummyRegressor: An unfitted regressor. Fit it with one row per training
            prefix and its remaining time as the target. It then predicts the
            arithmetic mean of those targets for every input row.
    """
    return DummyRegressor(strategy="mean")

