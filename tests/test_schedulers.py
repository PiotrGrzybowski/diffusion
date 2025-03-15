import torch

from diffusion.schedulers import LinearScheduler


def test_linear_scheduler():
    scheduler = LinearScheduler(1000, 0.02, 0.0001)

    assert torch.allclose(scheduler.schedule(), torch.linspace(0.02, 0.0001, 1000)), "LinearScheduler mismatch"
