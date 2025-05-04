# def scaled_dot_prduct_attention(query, key, values, ...) ->

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, n_heads=2, proj_dim=64, T=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.proj_dim = proj_dim
        self.q_linear = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.k_linear = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.v_linear = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        self.out_fc = nn.Linear(self.proj_dim, self.out_dim, bias=False)
        ## register mask
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    # Input - BXTXC
    def forward(self, q, k, v):
        B, T, C = q.size()  # Assuming k, v, q are same size
        q_proj = (
            self.q_linear(q)
            .view(B, T, self.n_heads, self.proj_dim // self.n_heads)
            .permute(0, 2, 1, 3)
        )  # proj_dim is divisible by n_heads
        k_proj = (
            self.k_linear(k)
            .view(B, T, self.n_heads, self.proj_dim // self.n_heads)
            .permute(0, 2, 1, 3)
        )
        v_proj = (
            self.v_linear(v)
            .view(B, T, self.n_heads, self.proj_dim // self.n_heads)
            .permute(0, 2, 1, 3)
        )

        # compute dot prodcut, sclae it
        attention = (q_proj @ k_proj.transpose(-1, -2)) / math.sqrt(self.proj_dim)
        attention = attention.masked_fill(self.mask == 0, float("-inf"))
        attention = nn.functional.softmax(attention, dim=-1)
        y = attention @ v_proj  # Bxn_headsxTxproj_dim
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, self.proj_dim)
        out = self.out_fc(y)
        return out, attention


# shape of x = B, T, T
def softmax(x):
    y = torch.exp(x / x.shape[-1]) / torch.sum(torch.exp(x / x.shape[-1]), dim=-1)
    return y


input = torch.randn(2, 8, 32)
self_attention = SelfAttention()
out, attention = self_attention(input, input, input)
print(f"output shape {out.shape} attention {attention}")
