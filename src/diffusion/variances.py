from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler


@dataclass(frozen=True)
class VarianceInputs:
    timesteps: torch.Tensor
    model_output: torch.Tensor | None = None


class VarianceStrategy(Protocol):
    def variance(self, inputs: VarianceInputs) -> torch.Tensor: ...

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor: ...


class FixedSmallVariance:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        betas = self.factors.betas[inputs.timesteps]
        gammas = self.factors.gammas[inputs.timesteps]
        gammas_prev = self.factors.gammas_prev[inputs.timesteps]

        return betas * (1 - gammas_prev) / (1 - gammas)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        timesteps = torch.where(inputs.timesteps == 0, torch.tensor(1), inputs.timesteps)
        inputs = VarianceInputs(timesteps, inputs.model_output)

        variance = self.variance(inputs)
        return torch.log(variance)


class FixedLargeVariance:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return self.factors.betas[inputs.timesteps]

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.log(self.variance(inputs))


class TrainableVariance:
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable variance")
        return torch.exp(inputs.model_output)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable variance")
        return inputs.model_output


class TrainableRangeVariance:
    def __init__(self, lower_variance: VarianceStrategy, upper_variance: VarianceStrategy) -> None:
        self.lower_variance = lower_variance
        self.uper_variance = upper_variance

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.exp(self.log_variance(inputs))

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable range variance")
        v = (inputs.model_output + 1) / 2
        return v * self.uper_variance.log_variance(inputs) + (1 - v) * self.lower_variance.log_variance(inputs)


class VarianceType(Enum):
    FIXED_SMALL = "FixedSmall"
    FIXED_LARGE = "FixedLarge"
    TRAINABLE = "Trainable"
    TRAINABLE_RANGE = "TrainableRange"


def fixed_small_variance(factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    betas = factors.betas[timesteps]
    gammas = factors.gammas[timesteps]
    gammas_prev = factors.gammas_prev[timesteps]

    return betas * (1 - gammas_prev) / (1 - gammas)


def fixed_small_log_variance(factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute the log of the variance at a given timestep.
    It handles the edge case where the timestep is 0 by replacing it with 1 to avoid log(0).
    Args:
        factors (Factors): The factors of the diffusion process.
        timesteps (torch.Tensor): The timesteps to compute at.
    Returns:
        torch.Tensor: The log of the variance at the given timesteps.
    """
    timesteps = torch.where(timesteps == 0, torch.tensor(1), timesteps)
    variance = fixed_small_variance(factors, timesteps)
    return torch.log(variance)


def fixed_large_variance(factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    """At timestep t=0, posterior_variance[1] is used instead of betas[0] to ensure numerical stability in log domain
    and better decoder initialization. This choice provides a smoother variance schedule and reduces noise for sharper reconstructions."""
    variance = factors.betas[timesteps]
    idx = torch.where(timesteps == 0)[0]
    small_variance = fixed_small_variance(factors, torch.tensor([1]))
    variance[idx] = small_variance

    return variance


def fixed_large_log_variance(factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    return torch.log(fixed_large_variance(factors, timesteps))


def trainable_variance(x: torch.Tensor) -> torch.Tensor:
    """Compute the variance from the input tensor, due to numerical stability model is trained in log space.
    Args:
        x (torch.Tensor): Predicted log variance.
    Returns:
        torch.Tensor: Predicted variance.
    """
    return torch.exp(x)


def trainable_log_variance(x: torch.Tensor) -> torch.Tensor:
    """Return the input tensor as is, since the model is trained in log space.
    Args:
        x (torch.Tensor): Predicted log variance.
    Returns:
        torch.Tensor: Predicted log variance.
    """
    return x


def trainable_range_log_variance(x: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute the log of the variance at a given timestep using model output as a range between small and large variance.
    Args:
        x (torch.Tensor): Model output tensor.
        factors (Factors): The factors of the diffusion process.
        timesteps (torch.Tensor): The timesteps to compute at.
    Returns:
        torch.Tensor: The log of the trainable range variance at the given timesteps.
    """
    v = (x + 1) / 2
    max_variance = fixed_large_log_variance(factors, timesteps)
    min_variance = fixed_small_log_variance(factors, timesteps)
    return v * max_variance + (1 - v) * min_variance


def trainable_range_variance(x: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute the variance at a given timestep using model output as a range between small and large variance.
    Args:
        x (torch.Tensor): Model output tensor.
        factors (Factors): The factors of the diffusion process.
        timesteps (torch.Tensor): The timesteps to compute at.
    Returns:
        torch.Tensor: The trainable range variance at the given timesteps.
    """
    return torch.exp(trainable_range_log_variance(x, factors, timesteps))


class VarianceManager:
    def __init__(self):
        self.variance_strategies = {
            VarianceType.FIXED_SMALL: (fixed_small_variance, fixed_small_log_variance),
            VarianceType.FIXED_LARGE: (fixed_large_variance, fixed_large_log_variance),
            VarianceType.TRAINABLE: (trainable_variance, trainable_log_variance),
            VarianceType.TRAINABLE_RANGE: (trainable_range_variance, trainable_range_log_variance),
        }

    def variance(self, strategy: VarianceType, timesteps: torch.Tensor, factors: Factors, x: torch.Tensor | None = None) -> torch.Tensor:
        return self._call_strategy(strategy, timesteps, factors, x, index=0)

    def log_variance(
        self, strategy: VarianceType, timesteps: torch.Tensor, factors: Factors, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._call_strategy(strategy, timesteps, factors, x, index=1)

    def _call_strategy(
        self, strategy: VarianceType, timestep: torch.Tensor, factors: Factors, x: torch.Tensor | None, index: int
    ) -> torch.Tensor:
        func = self.variance_strategies[strategy][index]
        if strategy in {VarianceType.FIXED_SMALL, VarianceType.FIXED_LARGE}:
            return func(factors, timestep)
        elif strategy == VarianceType.TRAINABLE and x is not None:
            return func(x)
        elif strategy == VarianceType.TRAINABLE_RANGE and x is not None:
            return func(x, factors, timestep)
        else:
            raise ValueError(f"Unknown variance strategy: {strategy}")


if __name__ == "__main__":
    scheduler = LinearScheduler(1000, 0.0001, 0.02)
    timesteps = torch.tensor([8, 900])
    factors = Factors(scheduler)
    torch.set_printoptions(precision=8)

    variance_old = fixed_small_variance(factors, timesteps)
    variance_new = FixedSmallVariance(factors).variance(VarianceInputs(timesteps, torch.empty(0)))

    result = torch.allclose(variance_old, variance_new)
    if not result:
        raise ValueError(f"Expected {variance_old}, but got {variance_new}")

    log_variance_old = fixed_small_log_variance(factors, timesteps)
    log_variance_new = FixedSmallVariance(factors).log_variance(VarianceInputs(timesteps, torch.empty(0)))
    result = torch.allclose(log_variance_old, log_variance_new)
    if not result:
        raise ValueError(f"Expected {log_variance_old}, but got {log_variance_new}")

    variance_old = fixed_large_variance(factors, timesteps)
    variance_new = FixedLargeVariance(factors).variance(VarianceInputs(timesteps, torch.empty(0)))
    result = torch.allclose(variance_old, variance_new)
    if not result:
        raise ValueError(f"Expected {variance_old}, but got {variance_new}")

    log_variance_old = fixed_large_log_variance(factors, timesteps)
    log_variance_new = FixedLargeVariance(factors).log_variance(VarianceInputs(timesteps, torch.empty(0)))
    result = torch.allclose(log_variance_old, log_variance_new)
    if not result:
        raise ValueError(f"Expected {log_variance_old}, but got {log_variance_new}")

    # prediction = torch.randn((2, 1, 2, 2))
    prediction = torch.tensor(
        [[[[-0.81177425, -1.56413877], [0.82801986, -0.39772224]]], [[[1.74248230, 0.52790087], [-0.64002794, -0.31274268]]]]
    )
    # prediction = torch.tensor([[[[1.43131948, -1.36195362], [1.07874191, 0.76396883]]]])
    variance_old = trainable_variance(prediction)
    variance_new = TrainableVariance().variance(VarianceInputs(torch.empty(0), prediction))
    result = torch.allclose(variance_old, variance_new)
    if not result:
        raise ValueError(f"Expected {variance_old}, but got {variance_new}")

    log_variance_old = trainable_log_variance(prediction)
    log_variance_new = TrainableVariance().log_variance(VarianceInputs(torch.empty(0), prediction))
    result = torch.allclose(log_variance_old, log_variance_new)
    if not result:
        raise ValueError(f"Expected {log_variance_old}, but got {log_variance_new}")

    variance_old = trainable_range_variance(prediction, factors, timesteps)
    variance_new = TrainableRangeVariance(FixedSmallVariance(factors), FixedLargeVariance(factors)).variance(
        VarianceInputs(timesteps, prediction)
    )
    print(variance_new)

    result = torch.allclose(variance_old, variance_new)
    if not result:
        raise ValueError(f"Expected {variance_old}, but got {variance_new}")

    log_variance_old = trainable_range_log_variance(prediction, factors, timesteps)
    log_variance_new = TrainableRangeVariance(FixedSmallVariance(factors), FixedLargeVariance(factors)).log_variance(
        VarianceInputs(timesteps, prediction)
    )
    result = torch.allclose(log_variance_old, log_variance_new)
    if not result:
        raise ValueError(f"Expected {log_variance_old}, but got {log_variance_new}")
