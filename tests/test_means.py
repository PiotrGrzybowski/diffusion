import pytest
import torch

from diffusion.diffusion_factors import Factors
from diffusion.means import DirectMean, EpsilonMean, MeanInputs, XStartMean, epsilon_from_xstart, x_start_from_mean
from diffusion.schedulers import LinearScheduler


torch.set_printoptions(precision=8)


@pytest.fixture
def inputs():
    betas = LinearScheduler(1000, 0.0001, 0.02).schedule()
    factors = Factors(betas)
    model_output = torch.tensor([[[[1.60896802]]], [[[1.50490487]]]])
    timesteps = torch.tensor([8, 900], dtype=torch.long)
    x_t = torch.tensor([[[[0.55517131]]], [[[0.62717044]]]])

    return MeanInputs(x_t, model_output, factors, timesteps)


def test_direct_mean_strategy(inputs: MeanInputs):
    strategy = DirectMean()
    mean_outputs = strategy.forward(inputs)
    mean = mean_outputs.mean
    target_mean = torch.tensor([1.60896802, 1.50490487])
    target_x_start = x_start_from_mean(inputs.x_t, inputs.model_output, inputs.factors, inputs.timesteps)
    target_epsilon = epsilon_from_xstart(target_x_start, inputs.x_t, inputs.factors, inputs.timesteps)

    target_x_start = torch.tensor([[[[7.12551117]]], [[[2953.06787109]]]])
    target_epsilon = torch.tensor([[[[-163.29997253]]], [[[-47.92512512]]]])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "DirectMean mismatch"
    assert torch.allclose(mean_outputs.x_start.squeeze(), target_x_start.squeeze()), "XStart mismatch"
    assert torch.allclose(mean_outputs.epsilon.squeeze(), target_epsilon.squeeze()), "Epsilon mismatch"


def test_x_start_mean_strategy(inputs: MeanInputs):
    strategy = XStartMean()
    mean_outputs = strategy.forward(inputs)

    mean = mean_outputs.mean
    x_start = mean_outputs.x_start
    epsilon = mean_outputs.epsilon

    target_mean = torch.tensor([[[[0.72419381]]], [[[0.62193859]]]])
    target_x_start = torch.tensor([[[[1.60896814]]], [[[1.50489962]]]])
    target_epsilon = torch.tensor([[[[-26.18181038]]], [[[0.60251254]]]])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "XStartMean mismatch"
    assert torch.allclose(epsilon.squeeze(), target_epsilon.squeeze()), "Epsilon mismatch"
    assert torch.allclose(x_start.squeeze(), target_x_start.squeeze()), "XStart mismatch"


def test_epsilon_mean_strategy(inputs: MeanInputs):
    strategy = EpsilonMean()
    mean_outputs = strategy.forward(inputs)
    mean = mean_outputs.mean

    target_mean = torch.tensor([[[[0.54486120]]], [[[0.60551941]]]])
    target_x_start = torch.tensor([[[[0.49088836]]], [[[-53.38068008]]]])
    target_epsilon = torch.tensor([[[[1.60896802]]], [[[1.50490487]]]])

    assert torch.allclose(mean.squeeze(), target_mean.squeeze()), "XStartMean mismatch"
    assert torch.allclose(mean_outputs.x_start.squeeze(), target_x_start.squeeze()), "XStart mismatch"
    assert torch.allclose(mean_outputs.epsilon.squeeze(), target_epsilon.squeeze()), "Epsilon mismatch"
