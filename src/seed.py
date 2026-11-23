import os
import random
from pathlib import Path

import numpy as np

from src.data_loading import load_config

DEFAULT_CONFIG = "config/pipeline.yaml"


def load_seed(path: str | Path = DEFAULT_CONFIG) -> int:
    """
    Read the random seed from a YAML configuration file.

    Args:
        path (str or Path): The configuration file with a top-level "seed" key.

    Returns:
        int: The seed.

    Raises:
        ValueError: If the file has no integer "seed".
    """
    seed = (load_config(path) or {}).get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path} must contain an integer 'seed'.")
    return seed


def set_seed(seed: int) -> None:
    """
    Seed every source of randomness used by the pipeline.

    This seeds Python's `random`, NumPy, and PYTHONHASHSEED (for subprocesses).
    Scikit-learn models must additionally receive `random_state=seed`.

    Args:
        seed (int): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
