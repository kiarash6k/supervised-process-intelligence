import random

import numpy as np
import pytest

from src.seed import load_seed, set_seed


def draw():
    return random.random(), float(np.random.rand())


def test_load_seed_reads_the_configuration(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")

    assert load_seed(path) == 7


def test_project_configuration_has_a_seed():
    assert isinstance(load_seed("config/pipeline.yaml"), int)


@pytest.mark.parametrize("content", ["name: demo\n", "seed: abc\n", "seed: true\n", ""])
def test_load_seed_rejects_a_missing_or_invalid_seed(tmp_path, content):
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        load_seed(path)


def test_same_seed_gives_the_same_random_numbers():
    set_seed(42)
    first = draw()
    set_seed(42)
    second = draw()

    assert first == second


def test_different_seeds_give_different_random_numbers():
    set_seed(1)
    first = draw()
    set_seed(2)
    second = draw()

    assert first != second
