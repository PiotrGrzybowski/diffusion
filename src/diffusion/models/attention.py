import torch
import torch.nn as nn

from diffusion.models.normalization import RMSNorm


class DotAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = head_dim**-0.5
        self.heads = heads
        self.head_dim = head_dim
        self.to_qkv = nn.Conv2d(dim, heads * head_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(heads * head_dim, dim, kernel_size=1)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = (
            x.view(b, self.heads, self.head_dim, h, w).permute(0, 1, 3, 4, 2).contiguous().view(b, self.heads, h * w, self.head_dim)
            for x in qkv
        )

        similarity = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = similarity.softmax(dim=-1)

        out = torch.matmul(attention, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.head_dim * self.heads, h, w)
        out = self.to_out(out)

        return out


class OptimizedAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = head_dim**-0.5
        self.heads = heads
        self.head_dim = head_dim
        self.norm = RMSNorm(dim)

        self.embed_dim = head_dim * heads
        self.projection = nn.Conv2d(dim, self.embed_dim, kernel_size=1, bias=False)
        self.final_projection = nn.Conv2d(self.embed_dim, dim, kernel_size=1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.norm(x)

        projected_input = self.projection(x)
        projected_input = projected_input.view(b, self.embed_dim, h * w).permute(0, 2, 1)
        attention_out, _ = self.multihead_attention(projected_input, projected_input, projected_input)
        attention_out = attention_out.permute(0, 2, 1).view(b, self.embed_dim, h, w)
        out = self.final_projection(attention_out)
        return out


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim
        self.hidden_dim = head_dim * heads

        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(self.hidden_dim, dim, kernel_size=1), RMSNorm(dim))

        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (x.view(b, self.heads, self.head_dim, h * w) for x in qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        contex = torch.matmul(k, v.transpose(-2, -1))
        q = q * self.scale

        out = torch.matmul(contex.transpose(-1, -2), q)
        out = out.view(b, self.hidden_dim, h, w)
        out = self.to_out(out)
        return out


def create_attention(attention: str, dim: int, heads: int, head_dim: int) -> nn.Module:
    if attention == "linear":
        return LinearAttention(dim, heads, head_dim)
    elif attention == "dot":
        return DotAttention(dim, heads, head_dim)
    else:
        raise ValueError(f"Unknown attention type {attention}")
