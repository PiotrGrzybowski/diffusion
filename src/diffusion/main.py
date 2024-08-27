import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop, Compose, Lambda, Resize, ToTensor


image_size = 512
transform = Compose(
    [
        ToTensor(),  # turn into torch Tensor of shape CHW, divide by 255
        Resize(image_size),
        CenterCrop(image_size),
        Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_transform = Compose(
    [
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.0),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
    ]
)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    print(a.shape)
    print(out.shape)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def main():
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    print(betas)
    print(alphas_cumprod.shape)
    alphas_cumprod_prev = torch.cat([torch.tensor([1]), alphas_cumprod[:-1]])

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    image = cv2.imread("image.png")
    print(image.shape)

    x_start = transform(image).unsqueeze(0)

    restored = reverse_transform(x_start.squeeze(0))
    # cv2.imshow('image', restored)
    # cv2.waitKey(0)

    def q_sample(x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # print(sqrt_alphas_cumprod_t.shape)
        # print(sqrt_one_minus_alphas_cumprod_t.shape)
        # print(sqrt_alphas_cumprod_t)
        # print(sqrt_one_minus_alphas_cumprod_t)
        # print(noise.shape)
        noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image

    noise_leves = torch.tensor([1, 10, 50, 100, 200, 500, 999])
    x_noisies = [reverse_transform(q_sample(x_start, torch.tensor([x])).squeeze(0)) for x in noise_leves]
    print(x_noisies[0].shape)
    merged = np.concatenate(x_noisies, axis=1)
    print(merged.shape)
    # x_noisy = q_sample(x_start, torch.tensor([999]))
    # restored = reverse_transform(x_noisy.squeeze(0))
    cv2.imshow("image", merged)
    cv2.waitKey(0)


main()
