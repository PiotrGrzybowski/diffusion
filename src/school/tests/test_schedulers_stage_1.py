import torch
from school.schedulers import LinearScheduler


def test_linear_scheduler_initialization():
    """Test the initialization of the LinearScheduler class."""
    timesteps = 1000
    start = 0.0001
    end = 0.02

    scheduler = LinearScheduler(timesteps, start, end)

    assert scheduler.timesteps == timesteps, "Incorrect number of timesteps."
    assert scheduler.start == start, "Incorrect start value."
    assert scheduler.end == end, "Incorrect end value."


def test_linear_scheduler_schedule_shape():
    """Test the shape of the scheduled tensor."""
    timesteps = 1000
    start = 0.0001
    end = 0.02

    scheduler = LinearScheduler(timesteps, start, end)
    betas = scheduler.schedule()

    assert betas.shape == torch.Size([timesteps]), f"Expected schedule shape {[timesteps]}, got {betas.shape}."


def test_linear_scheduler_schedule_values():
    """Test the correctness of the scheduled values."""
    timesteps = 10
    start = 0.1
    end = 1.0

    scheduler = LinearScheduler(timesteps, start, end)
    betas = scheduler.schedule()
    expected_schedule = torch.linspace(start, end, timesteps)

    assert torch.allclose(betas, expected_schedule), "Scheduled values are incorrect."


def test_linear_scheduler_edge_cases():
    """Test edge cases for the LinearScheduler."""
    timesteps = 1
    start = 0.5
    end = 0.5

    scheduler = LinearScheduler(timesteps, start, end)
    betas = scheduler.schedule()

    assert betas.shape == torch.Size([1]), "Schedule shape for single timestep is incorrect."
    assert torch.allclose(betas, torch.tensor([0.5])), "Schedule values for single timestep are incorrect."

    timesteps = 10
    start = 0.1
    end = 0.1

    scheduler = LinearScheduler(timesteps, start, end)
    betas = scheduler.schedule()

    assert torch.allclose(betas, torch.full((timesteps,), 0.1)), "Schedule values for constant start and end are incorrect."
