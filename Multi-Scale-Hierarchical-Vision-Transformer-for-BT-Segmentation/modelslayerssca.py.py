import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA(nn.Module):
    """
    Scale-Conditional Attention
    Returns both output features and attention probabilities
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, scale_emb):
        """
        x: (B, N, C)
        scale_emb: (1, 1, C)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Inject scale embedding into keys
        scale_emb = scale_emb.expand(B, N, C)
        scale_emb = scale_emb.view(B, N, self.num_heads, self.head_dim)
        scale_emb = scale_emb.permute(0, 2, 1, 3)

        k = k + scale_emb

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_logits, dim=-1)

        out = attn_probs @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out, attn_probs
