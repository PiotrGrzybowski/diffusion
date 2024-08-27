import torch
import torch.nn as nn
from einops.einops import rearrange

from diffusion.normalization import RMSNorm


class Attention(nn.Module):
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
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        q1, k1, v1 = (rearrange(x, "b (h c) x y -> b h (x y) c", h=self.heads) for x in qkv)

        q, k, v = (
            x.view(b, self.heads, self.head_dim, h, w).permute(0, 1, 3, 4, 2).contiguous().view(b, self.heads, h * w, self.head_dim)
            for x in qkv
        )

        assert torch.allclose(q1, q)

        similarity = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # print(similarity.shape, sim.shape)
        assert torch.allclose(similarity, sim)

        attention = similarity.softmax(dim=-1)

        out = torch.matmul(attention, v)
        out1 = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        # print(out.shape, out1.shape)

        assert torch.allclose(out, out1)

        out_last = out.permute(0, 1, 3, 2).contiguous().view(b, self.head_dim * self.heads, h, w)
        out_last_1 = rearrange(out, "b h (x y) c -> b (h c) x y", x=h)

        assert torch.allclose(out_last, out_last_1)

        out = self.to_out(out_last)

        return out


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim
        self.hidden_dim = head_dim * heads

        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, kernel_size=1, bias=False)
        # self.to_out = nn.Sequential(nn.Conv2d(self.hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim))
        self.to_out = nn.Sequential(nn.Conv2d(self.hidden_dim, dim, kernel_size=1), RMSNorm(dim))

        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (x.view(b, self.heads, self.head_dim, h * w) for x in qkv)
        q1, k1, v1 = (rearrange(x, "b (h c) x y -> b h c (x y)", h=self.heads) for x in qkv)

        # print(q.shape, q1.shape)
        # print(k.shape, k1.shape)

        assert torch.allclose(q, q1)
        assert torch.allclose(k, k1)
        assert torch.allclose(v, v1)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q1 = q1.softmax(dim=-2)
        k1 = k1.softmax(dim=-1)

        assert torch.allclose(q, q1)
        assert torch.allclose(k, k1)

        # q = q * self.scale
        # print(q.shape, k.shape)
        #
        context = torch.einsum("b h d n, b h e n -> b h d e", k1, v1)
        contex = torch.matmul(k, v.transpose(-2, -1))
        q = q * self.scale
        q1 = q1 * self.scale

        assert torch.allclose(context, contex)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q1)
        ou = torch.matmul(contex.transpose(-1, -2), q)

        # print(out.shape, ou.shape)
        assert torch.allclose(out, ou)

        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=h)
        ou = ou.view(b, self.hidden_dim, h, w)

        assert torch.allclose(out, ou)

        out = self.to_out(out)
        ou = self.to_out(ou)

        assert torch.allclose(out, ou)

        return ou


def create_attention(attention: str, dim: int, heads: int, head_dim: int) -> nn.Module:
    if attention == "linear":
        return LinearAttention(dim, heads, head_dim)
    elif attention == "dot":
        return Attention(dim, heads, head_dim)
    else:
        raise ValueError(f"Unknown attention type {attention}")


if __name__ == "__main__":
    x = torch.rand(4, 256, 16, 16)

    attention = Attention(256, 4, 300)
    print(attention)
    out = attention(x)

    linear_attention = LinearAttention(256, 4, 300)
    out = linear_attention(x)
    print(out.shape)
