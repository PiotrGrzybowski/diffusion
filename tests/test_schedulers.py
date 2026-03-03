import torch

from diffusion.schedulers import CosineScheduler, LinearScheduler


def test_linear_scheduler():
    scheduler = LinearScheduler(1000, 0.02, 0.0001)

    assert torch.allclose(scheduler.schedule(), torch.linspace(0.02, 0.0001, 1000)), "LinearScheduler mismatch"


def test_cosine_scheduler():
    scheduler = CosineScheduler(1000, s=0.008, max_beta=0.999)
    betas = scheduler.schedule()

    assert betas.shape == (1000,)
    assert torch.all(betas > 0)
    assert torch.all(betas <= 0.999)
