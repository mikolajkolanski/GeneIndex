import torch
from torch import nn
import torch.nn.functional as F

class Att(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super(Att, self).__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.proj = nn.Linear(d_model, d_model)
 
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

def window_partition(x, window_size):
    """[B, H, W, C] -> [B * num_windows, window_size*window_size, C]"""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows, (Hp, Wp)
 
 
def window_unpartition(windows, window_size, pad_hw, hw):
    """Reverse of window_partition."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
    C = windows.shape[-1]
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x
 
 
class Block(nn.Module):

    def __init__(self, d_model, num_heads=8, window_size=0):
        super().__init__()
        self.window_size = window_size
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Att(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )
 
    def forward(self, x, H, W):
        B, N, C = x.shape
        use_window = self.window_size > 0 and N == H * W
 
        shortcut = x
        x = self.ln1(x)
 
        if use_window:
            x = x.view(B, H, W, C)
            x, pad_hw = window_partition(x, self.window_size)
 
        x = self.attn(x)
 
        if use_window:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            x = x.reshape(B, N, C)
 
        x = shortcut + x
        x = x + self.mlp(self.ln2(x))
        return x
