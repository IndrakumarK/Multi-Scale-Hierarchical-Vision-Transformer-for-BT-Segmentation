import torch
import torch.nn as nn


class HATD(nn.Module):
    """
    Entropy-based Hierarchy-Aware Token Decimation
    """
    def __init__(self, tau=1.5, min_tokens=32):
        super().__init__()
        self.tau = tau
        self.min_tokens = min_tokens

    def forward(self, tokens, attn_probs):
        """
        tokens: (B, N, C)
        attn_probs: (B, H, N, N)
        """
        B, N, C = tokens.shape

        # Average attention over heads
        p = attn_probs.mean(dim=1)  # (B, N, N)

        # Shannon entropy
        entropy = -torch.sum(p * torch.log(p + 1e-6), dim=-1)  # (B, N)

        kept_tokens = []

        for b in range(B):
            mask = entropy[b] > self.tau
            t = tokens[b][mask]

            # Safety: ensure minimum tokens
            if t.shape[0] < self.min_tokens:
                topk = torch.topk(entropy[b], self.min_tokens).indices
                t = tokens[b][topk]

            kept_tokens.append(t)

        max_len = max(t.shape[0] for t in kept_tokens)
        out = tokens.new_zeros((B, max_len, C))

        for b, t in enumerate(kept_tokens):
            out[b, :t.shape[0]] = t

        return out
