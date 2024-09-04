import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from diffusion.unet import FocusDownsample


class EinopsDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.rearrange(x)
        out = self.conv(out)
        return out


x = torch.rand(16, 32, 256, 256)

f = FocusDownsample(32, 32)
e = EinopsDownsample(32, 32)

# y = r(x.clone())
# print(y.shape)
# zz = f(x.clone())
# print(zz.shape)

# assert torch.allclose(y, zz)
e.conv.weight.data = f.conv.weight.data
e.conv.bias.data = f.conv.bias.data

assert torch.allclose(f(x), e(x))
