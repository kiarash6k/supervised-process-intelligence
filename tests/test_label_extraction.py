
import pandas as pd
import pytest

from data_loading import load_dataset
from labels import extract_labels


def test_extract_labels(tmp_path):
    log = load_dataset("config/helpdesk.yaml")
    print(log.columns)