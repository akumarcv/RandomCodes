import torch 
import torch.nn as nn
import math 

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block_size = 8
        self.n_heads = 8
        self.in_dim = 64
        self.embed_dim = 64
        self.out_dim = 64
        self.q_linear_in = nn.Linear(self.in_dim, 3*self.embed_dim, bias=False)
        self.q_linear_out = nn.Linear(self.embed_dim, self.out_dim, bias = False)
        
        self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))
        
    def forward(self, x):
        B, T, C = x.size()
        in_proj = self.q_linear_in(x)
        q, k, v = in_proj.split(self.embed_dim, dim=2)
        
        q = q.view(B, T, self.n_heads, self.embed_dim//self.n_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, self.embed_dim//self.n_heads).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_heads, self.embed_dim//self.n_heads).permute(0, 2, 1, 3)
        
        attention = (q @ k.transpose(-1, -2) )/math.sqrt(k.shape[-1])
        attention = attention.masked_fill(self.mask==0, float("-inf"))
        attention = nn.functional.softmax(attention, dim = -1)
        y = attention @ v # (B x nh x Tx T) X (B x nh x T x C) -> (B x nh x T X C)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        y = self.q_linear_out(y)
        # print(f"in_proj {in_proj.shape} q {q.shape} k {k.shape} v {v.shape} A {attention.shape} y {y.shape}")
        return y
        

class MLP(nn.Module):
    def __init__(self, in_embed, out_embed):
        super().__init__()
        self.l1 = nn.Linear(in_embed, 4 * in_embed, bias=False)
        self.g1 = nn.GELU()
        self.l2 = nn.Linear(4 * in_embed, in_embed)
        self.d1 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.g1(x)
        x = self.l2(x)
        x = self.d1(x)
        return x

class Block(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mha1 = SelfAttention()
        self.ln1 = nn.LayerNorm(64)
        self.mha2 = SelfAttention()
        self.ln2 = nn.LayerNorm(64)
        self.linear = MLP(64, 64)
        self.ln3 = nn.LayerNorm(64)
    
    def forward(self, x):
        x = x + self.mha1(self.ln1(x))
        x = x + self.mha2(self.ln2(x))
        x = self.linear(self.ln3(x))
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_blocks = 3
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)]) 
    
    def forward(self, x):
        for i, m in enumerate(self.blocks):
            y = m(x)
        return y
        
            
decoder = Decoder()
x = torch.randn(2, 8, 64)
out = decoder(x)
print(f"out {out.shape} {out[0]}")
    