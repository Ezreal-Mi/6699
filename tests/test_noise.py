import torch
import torch.nn as nn

from src.noise import add_input_noise, fgsm_attack, pgd_attack


def test_add_input_noise_sigma_zero_clones_input():
    x = torch.linspace(0, 1, 5).view(-1, 1)
    noisy = add_input_noise(x, sigma=0.0)

    assert torch.equal(noisy, x)
    assert noisy.data_ptr() != x.data_ptr()


def test_attacks_keep_points_inside_unit_interval():
    model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.linspace(0, 1, 8).view(-1, 1)
    y = torch.zeros_like(x)

    fgsm = fgsm_attack(model, x, y, epsilon=0.2)
    pgd = pgd_attack(model, x, y, epsilon=0.2, n_steps=2, n_restarts=1, seed=0)

    assert torch.all(fgsm >= 0)
    assert torch.all(fgsm <= 1)
    assert torch.all(pgd >= 0)
    assert torch.all(pgd <= 1)
    assert torch.all((pgd - x).abs() <= 0.200001)
