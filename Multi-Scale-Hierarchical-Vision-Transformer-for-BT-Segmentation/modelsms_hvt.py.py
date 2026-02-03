import torch
import torch.nn as nn
import torch.nn.functional as F

from modelslayerssca import SCA
from modelslayershatd import HATD
from modelslayersagf import AGF


class MSHVT(nn.Module):
    """
    Multi-Scale Hierarchical Vision Transformer (MS-HVT)
    """
    def __init__(self, embed_dims=[64, 128, 256], num_heads=[2, 4, 8]):
        super().__init__()

        self.num_scales = len(embed_dims)

        # Scale embeddings
        self.scale_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, d)) for d in embed_dims
        ])

        # Scale-conditional attention blocks
        self.sca_blocks = nn.ModuleList([
            SCA(embed_dims[i], num_heads[i])
            for i in range(self.num_scales)
        ])

        # Hierarchy-aware token decimation
        self.hatd = HATD(tau=1.5, min_tokens=32)

        # Adaptive gated fusion
        self.agf = AGF(embed_dims)

    def forward(self, feats):
        """
        feats: list of tensors [F1, F2, F3]
               each Fi shape = (B, Ni, Ci)
        """
        refined_feats = []

        for i in range(self.num_scales):
            feat = feats[i]
            scale_emb = self.scale_embeds[i]

            feat, attn_probs = self.sca_blocks[i](feat, scale_emb)
            feat = self.hatd(feat, attn_probs)

            refined_feats.append(feat)

        fused = self.agf(refined_feats)
        return fused
