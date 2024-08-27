import torch
import torch.nn as nn

from diffusion.models.attention import create_attention
from diffusion.models.normalization import RMSNorm
from diffusion.models.time_embedding import TimeEmbedding


class FocusDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(2, 2), stride=2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.unfold(x)
        x = x.view(b, c * 4, h // 2, w // 2)
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = RMSNorm(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int | None = None, dropout: float = 0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2)) if time_dim else None
        self.block1 = Block(in_channels, out_channels, dropout)
        self.block2 = Block(out_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_embedding is not None:
            scale_shift = self.mlp(time_embedding)
            scale_shift = scale_shift.unsqueeze(-1).unsqueeze(-1)
            scale_shift = scale_shift.chunk(2, dim=1)

        out = self.block1(x, scale_shift)
        out = self.block2(out)

        return out + self.residual_conv(x)


class EncoderSegment(nn.Module):
    def __init__(
        self,
        block_channels: int,
        downsample_channels: int,
        time_dim: int,
        attention: str,
        heads: int,
        head_dim: int,
        downsample: bool,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block1 = ResnetBlock(block_channels, block_channels, time_dim, dropout)
        self.block2 = ResnetBlock(block_channels, block_channels, time_dim, dropout)
        self.attention = create_attention(attention, block_channels, heads, head_dim)
        self.downsample = FocusDownsample(block_channels, downsample_channels) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        residuals = []

        x = self.block1(x, time)
        residuals.append(x)

        x = self.block2(x, time)
        x = self.attention(x) + x
        residuals.append(x)

        x = self.downsample(x)

        return x, residuals


class UnetEncoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        downsamples: list[bool],
        time_dim: int,
        attentions: list[str],
        heads: int,
        head_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.segments = nn.ModuleList()

        for block_channels, downsample_channels, downsample, attention in zip(channels[:-1], channels[1:], downsamples, attentions):
            self.segments.append(
                EncoderSegment(block_channels, downsample_channels, time_dim, attention, heads, head_dim, downsample, dropout)
            )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        residuals = []
        for segment in self.segments:
            x, residual = segment(x, time)
            residuals.append(residual)
        return x, residuals


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_channels: int,
        heads: int,
        head_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        time_dim = init_channels * 4
        self.time_embedding = TimeEmbedding(init_channels, time_dim)
        self.init_conv = nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1)
        self.final_block = ResnetBlock(init_channels * 2, init_channels, time_dim, dropout)
        self.final_conv = nn.Conv2d(init_channels, in_channels, kernel_size=1)

        self.encoder = UnetEncoder(
            [128, 128, 256, 512], [True, True, False], time_dim, ["linear", "linear", "dot"], heads, head_dim, dropout
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time = self.time_embedding(timesteps)
        print(f"Time embedding shape: {time.shape}")
        x = self.init_conv(x)
        x, residuals = self.encoder(x, time)

        x = self.final_block(x, time)
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    encoder = UnetEncoder([128, 128, 256, 512], [True, True, False], 512, ["linear", "linear", "dot"], 4, 32, 0)
    time = torch.rand((16, 512))
    # print(encoder)
    x = torch.rand((16, 128, 64, 64))
    encoder(x, time)
