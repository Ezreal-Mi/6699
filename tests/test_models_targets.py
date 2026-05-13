import torch

from src.models import ModelBuilder, count_parameters, matched_shallow_width
from src.targets import make_dataset, sawtooth_target


def test_phase1_rescue_parameter_match():
    deep = ModelBuilder(depth=9, width=16)

    assert count_parameters(deep) == 1953
    assert matched_shallow_width(1953) == 651
    assert count_parameters(ModelBuilder(depth=2, width=651)) == 1954


def test_sawtooth_target_shape_and_range():
    x, y = make_dataset(n=17, k=4)

    assert x.shape == (17, 1)
    assert y.shape == (17, 1)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)


def test_sawtooth_target_known_values_for_k1():
    x = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    expected = torch.tensor([[1.0], [0.5], [0.0], [0.5], [1.0]])

    assert torch.allclose(sawtooth_target(x, k=1), expected)
