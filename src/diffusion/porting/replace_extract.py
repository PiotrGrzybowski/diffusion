import cv2
import numpy as np
import torch
import torch.nn.functional as F
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


timesteps = 300
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev_old = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

assert torch.allclose(alphas_cumprod_prev, alphas_cumprod_prev_old)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


batch_size = 4
t = torch.randint(0, timesteps, (batch_size,))

image = cv2.imread("image.png")
# cv2.imshow('image', image)
# cv2.waitKey(0)
print(t)

img = transform(image)
batch = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)
print(batch.shape)


###q_sample
x_start = batch

noise = torch.randn_like(x_start)
sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
print(sqrt_alphas_cumprod_t)
print(sqrt_one_minus_alphas_cumprod_t)

q_sampled = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

print("sqrt(gamma_t) * x_t")
print(sqrt_alphas_cumprod_t.shape)
print(x_start.shape)
squeezed = sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
print(squeezed.shape)

assert torch.allclose(sqrt_alphas_cumprod_t * x_start, squeezed * x_start)
