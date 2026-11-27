"""Models for predicting the remaining time of case prefixes."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


def build_mean_predictor() -> DummyRegressor:
    """
    Build the mean remaining-time baseline (REQ-F25).

    Returns:
        DummyRegressor: An unfitted regressor. Fit it with one row per training
            prefix and its remaining time as the target. It then predicts the
            arithmetic mean of those targets for every input row.
    """
    return DummyRegressor(strategy="mean")

def build_ridge_regressor(*, alpha: float, categorical_attributes=(), numeric_attributes=(), include_reassignment: bool = False) -> Pipeline:
    """
    Build a Ridge pipeline for raw prefix feature tables.

    Args:
        alpha: L2 penalty strength from the pipeline configuration.
        categorical_attributes: Static fields to one-hot encode.
        numeric_attributes: Additional numeric static fields to scale.
        include_reassignment: Include the configured reassignment count.

    Returns:
        Pipeline: Fit on training features and remaining-time targets, then
            call predict on feature tables with the same configured fields.
    """
    preprocess = build_preprocessor(
        categorical_attributes=categorical_attributes,
        numeric_attributes=numeric_attributes,
        include_reassignment=include_reassignment,
        scale_numeric=True,
    )
    
    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", Ridge(alpha=alpha)),
    ])