import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

class EfficientProbing(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_queries = num_queries
        
        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        
        return x_cls