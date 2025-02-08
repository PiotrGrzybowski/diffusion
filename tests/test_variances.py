import pytest
import torch
from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler
from diffusion.variances import DirectVariance, FixedLargeVariance, FixedSmallVariance, TrainableRangeVariance, VarianceInputs


@pytest.fixture
def inputs():
    factors = Factors(LinearScheduler(1000, 0.0001, 0.02))
    model_output = torch.tensor(
        [[[[-0.81177425, -1.56413877], [0.82801986, -0.39772224]]], [[[1.74248230, 0.52790087], [-0.64002794, -0.31274268]]]]
    )
    timesteps = torch.tensor([8, 900], dtype=torch.long)

    return VarianceInputs(factors, timesteps, model_output)


def test_fixed_small_strategy(inputs: VarianceInputs):
    strategy = FixedSmallVariance()

    variance = strategy.variance(inputs)
    log_variance = strategy.log_variance(inputs)

    target_variance = torch.tensor([0.00021779, 0.01802784])
    target_log_variance = torch.log(target_variance)

    assert torch.allclose(variance.squeeze(), target_variance), "Fixed small variance mismatch"
    assert torch.allclose(log_variance.squeeze(), target_log_variance), "Fixed small log variance mismatch"


def test_fixed_large_strategy(inputs: VarianceInputs):
    strategy = FixedLargeVariance()

    variance = strategy.variance(inputs)
    log_variance = strategy.log_variance(inputs)

    target_variance = torch.tensor([0.00025936, 0.01802793])
    target_log_variance = torch.log(target_variance)

    assert torch.allclose(variance.squeeze(), target_variance), "Fixed large variance mismatch"
    assert torch.allclose(log_variance.squeeze(), target_log_variance), "Fixed large log variance mismatch"


def test_trainable_strategy(inputs: VarianceInputs):
    strategy = DirectVariance()

    variance = strategy.variance(inputs)
    log_variance = strategy.log_variance(inputs)

    target_variance = torch.tensor(
        [[[[0.44406947, 0.20926817], [2.28878212, 0.67184860]]], [[[5.71150351, 1.69536972], [0.52727771, 0.73143810]]]]
    )
    target_log_variance = torch.log(target_variance)

    assert torch.allclose(variance, target_variance), "Trainable variance mismatch"
    assert torch.allclose(log_variance, target_log_variance), "Trainable log variance mismatch"


def test_trainable_range_strategy(inputs: VarianceInputs):
    strategy = TrainableRangeVariance(FixedSmallVariance(), FixedLargeVariance())

    variance = strategy.variance(inputs)
    log_variance = strategy.log_variance(inputs)

    target_variance = torch.tensor(
        [[[[0.00022140, 0.00020732], [0.00025549, 0.00022956]]], [[[0.01802796, 0.01802791], [0.01802785, 0.01802786]]]]
    )
    target_log_variance = torch.log(target_variance)

    assert torch.allclose(variance, target_variance), "Trainable range variance mismatch"
    assert torch.allclose(log_variance, target_log_variance), "Trainable range log variance mismatch"
