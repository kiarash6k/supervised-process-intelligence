import pytest

from src.data_loading import load_dataset

SYNTHETIC_CONFIG = "tests/data/synthetic.yaml"


@pytest.fixture
def synthetic_log():
    """The normalized synthetic test log (4 cases, 13 events)."""
    return load_dataset(SYNTHETIC_CONFIG)
