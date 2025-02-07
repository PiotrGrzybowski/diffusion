import pytest
import torch
from diffusion.diffusion_factors import Factors
from diffusion.means import DirectMean, EpsilonMean, MeanInputs, XStartMean
from diffusion.schedulers import LinearScheduler


@pytest.fixture
def factors():
    return Factors(LinearScheduler(1000, 0.0001, 0.02))


@pytest.fixture
def inputs():
    model_output = torch.tensor([[[[1.60896802]]], [[[1.50490487]]]])
    timesteps = torch.tensor([8, 900], dtype=torch.long)
    x_t = torch.tensor([[[[0.55517131]]], [[[0.62717044]]]])

    return MeanInputs(timesteps, model_output, x_t)


def test_direct_mean_strategy(inputs: MeanInputs):
    strategy = DirectMean()
    mean = strategy.mean(inputs)
    target_mean = torch.tensor([1.60896802, 1.50490487])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "DirectMean mismatch"


def test_x_0_mean_strategy(factors: Factors, inputs: MeanInputs):
    strategy = XStartMean(factors)
    mean = strategy.mean(inputs)
    target_mean = torch.tensor([[[[0.72419381]]], [[[0.62193859]]]])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "XStartMean mismatch"


def test_epsilon_mean_strategy(factors: Factors, inputs: MeanInputs):
    strategy = EpsilonMean(factors)
    mean = strategy.mean(inputs)
    target_mean = torch.tensor([[[[0.54486120]]], [[[0.60551941]]]])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "XStartMean mismatch"
