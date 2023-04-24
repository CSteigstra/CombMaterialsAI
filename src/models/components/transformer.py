import math
from typing import Optional, Union, Callable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
from torch import Tensor

# class SimpleViT2(nn.Module):
    


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, key_padding_mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if key_padding_mask is not None:
            dots = dots.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, key_padding_mask = None):
        for attn, ff in self.layers:
            x = attn(x, key_padding_mask = key_padding_mask) + x
            x = ff(x) + x
        return x


class SimpleViT(nn.Module):
    def __init__(self, *,
                 image_size: int,
                 patch_size: int = 2,
                 num_classes: int = 1,
                 dim: int = 32,
                 depth: int = 1,
                 heads: int = 8,
                 mlp_dim: int = 128,
                 channels: int = 1,
                 dim_head: int = 64,
                 posemb: nn.Module):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(channels, dim, (patch_height, patch_width), (patch_height, patch_width)),
        #     nn.LayerNorm(dim)
        # ) #TODO: Implement convolutional embeddings

        self.posemb = posemb

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, key_padding_mask = None):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = self.posemb(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        if key_padding_mask is not None:
            key_padding_mask = rearrange(key_padding_mask, 'b ... -> b (...)')

        x = self.transformer(x, key_padding_mask = key_padding_mask)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)