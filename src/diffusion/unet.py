import math

import torch
import torch.nn as nn

from diffusion.attention import AttentionType, create_attention
from diffusion.normalization import RMSNorm


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, size: int, theta: float = 10000) -> None:
        super().__init__()
        self.size = size
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = math.log(self.theta) / (self.size // 2 - 1)
        embedding = torch.exp(torch.arange(self.size // 2).to(x) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return embedding


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


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
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

        if self.mlp is not None:
            print("################")
            print(self.mlp[1].weight.device)
            print(self.block1.conv.weight.device)
            print("################")

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_embedding is not None:
            scale_shift = self.mlp(time_embedding)
            scale_shift = scale_shift.unsqueeze(-1).unsqueeze(-1)
            scale_shift = scale_shift.chunk(2, dim=1)

        out = self.block1(x, scale_shift)
        out = self.block2(out)

        return out + self.residual_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        init_channels: int,
        channels: list[int],
        heads: int,
        head_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        # self.init_conv = nn.Conv2d(in_channels, init_channels, kernel_size=7, padding=3)
        time_dim = init_channels * 4
        self.init_conv = nn.Conv2d(in_channels, init_channels, kernel_size=3, padding=1)
        self.final_block = ResnetBlock(init_channels * 2, init_channels, time_dim, dropout)
        self.final_conv = nn.Conv2d(init_channels, in_channels, kernel_size=1)

        self.encoder = UnetEncoder(channels, time_dim, dropout, heads, head_dim)
        self.bottleneck = UnetBottleneck(channels[-1], channels[-1], time_dim, dropout, heads, head_dim)
        self.decoder = UnetDecoder(list(reversed(channels)), time_dim, dropout, heads, head_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(init_channels),
            nn.Linear(init_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(t)
        x = self.init_conv(x)
        residual = x
        x, skip_connections = self.encoder(x, t)
        x = self.bottleneck(x, t)
        x = self.decoder(x, reversed(skip_connections), t)

        x = torch.cat((x, residual), dim=1)
        x = self.final_block(x, t)
        x = self.final_conv(x)

        return x


class EncoderSegment(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float, attention_type: AttentionType, heads: int, head_dim: int
    ) -> None:
        super().__init__()
        self.block1 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.block2 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.attention = create_attention(attention_type, out_channels, heads, head_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        outputs = []
        x = self.block1(x, t)
        outputs.append(x)

        x = self.block2(x, t)

        x = self.attention(x) + x

        outputs.append(x)

        return outputs


class DecoderSegment(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float, attention_type: AttentionType, heads: int, head_dim: int
    ) -> None:
        super().__init__()
        self.block1 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.block2 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.attention = create_attention(attention_type, out_channels, heads, head_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.block1(x, t)

        x = torch.cat((x, skip_connections.pop()), dim=1)
        x = self.block2(x, t)

        x = self.attention(x) + x

        return x


class UnetEncoder(nn.Module):
    def __init__(self, channels: list[int], time_dim: int, dropout: float, heads: int, head_dim: int) -> None:
        super().__init__()
        self.segments = self._create_segments(channels[:-1], time_dim, dropout, heads, head_dim)
        self.post_segments = self._create_post_segments(channels)

    def _create_segments(self, channels: list[int], time_dim: int, dropout: int, heads: int, head_dim: int) -> nn.ModuleList:
        result = nn.ModuleList()
        attention_types = self._create_attentions_types(len(channels))

        for segment_channels, attention_type in zip(channels, attention_types):
            result.append(EncoderSegment(segment_channels, segment_channels, time_dim, dropout, attention_type, heads, head_dim))
        return result

    def _create_attentions_types(self, count: int) -> list[AttentionType]:
        types = [AttentionType.LINEAR] * count
        types[-1] = AttentionType.STANDARD
        return types

    def _create_post_segments(self, channels: list[int]) -> nn.ModuleList:
        result = nn.ModuleList()
        downsample_channels = channels[:-1]
        for in_channels, out_channels in zip(downsample_channels, downsample_channels[1:]):
            result.append(FocusDownsample(in_channels, out_channels))

        pre_bottleneck = nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1)
        result.append(pre_bottleneck)
        return result

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> list[list[torch.Tensor]]:
        skip_connections = []
        for segment, post_segment in zip(self.segments, self.post_segments):
            x = segment(x, t)
            skip_connections.append(x)
            x = x[-1]
            x = post_segment(x)
        return x, skip_connections


class UnetDecoder(nn.Module):
    def __init__(self, channels: list[int], time_dim: int, dropout: float, heads: int, head_dim: int) -> None:
        super().__init__()
        self.segments = self._create_segments(channels, time_dim, dropout, heads, head_dim)
        self.post_segments = self._create_post_segments(channels)

    def _create_segments(self, channels: list[int], time_dim: int, dropout: float, heads: int, head_dim: int) -> nn.ModuleList:
        result = nn.ModuleList()
        attention_types = self._create_attentions_types(len(channels))
        for first_channels, second_channels, attention_type in zip(channels, channels[1:], attention_types):
            result.append(
                DecoderSegment(first_channels + second_channels, first_channels, time_dim, dropout, attention_type, heads, head_dim)
            )
        return result

    def _create_post_segments(self, channels: list[int]) -> nn.ModuleList:
        result = nn.ModuleList()
        downsample_channels = channels[:-1]
        for in_channels, out_channels in zip(downsample_channels, downsample_channels[1:]):
            result.append(Upsample(in_channels, out_channels))

        post_last_segment = nn.Conv2d(channels[-2], channels[-1], kernel_size=3, padding=1)
        result.append(post_last_segment)
        return result

    def _create_attentions_types(self, count: int) -> list[AttentionType]:
        types = [AttentionType.LINEAR] * count
        types[0] = AttentionType.STANDARD
        return types

    def forward(self, x: torch.Tensor, skip_connections: list[list[torch.Tensor]], t: torch.Tensor) -> torch.Tensor:
        for segment, post_segment, connections in zip(self.segments, self.post_segments, skip_connections):
            x = segment(x, t, connections)
            x = post_segment(x)
        return x


class UnetBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float, heads: int, head_dim: int) -> None:
        super().__init__()
        self.block1 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.block2 = ResnetBlock(in_channels, out_channels, time_dim, dropout)
        self.attention = create_attention(AttentionType.STANDARD, out_channels, heads, head_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, t)
        x = self.attention(x) + x
        x = self.block2(x, t)
        return x


if __name__ == "__main__":
    # in_channels = 1
    # init_channels = 128
    # dim_mults = (1, 2, 4)
    # unet = Unet(in_channels, init_channels, dim_mults)
    # print(proper_config(in_channels, init_channels, [1, 2, 4]))
    # encoder_features = [128, 128, 256, 512]
    # encoder_block_features = encoder_features[1:]
    # encoder_downsample_features = encoder_features[1:]
    # encoder = UnetEncoder([128, 128, 256, 512])
    # print(encoder)
    # x = torch.rand(4, 128, 64, 64)
    # x, outputs = encoder(x, None)
    # # print(x.shape)
    # decoder = UnetDecoder([512, 256, 128, 128])
    # x = decoder(x, reversed(outputs), None)
    unet = Unet(1, 128, [128, 128, 256, 512], heads=4, head_dim=32)
    x = torch.rand(4, 1, 64, 64)
    t = torch.randint(0, 100, (4,))
    # print(t)
    out = unet(x, t)

    # # calculate number of parameters
    # import numpy as np
    #
    # model_parameters = filter(lambda p: p.requires_grad, unet.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
