import torch
import torch.nn as nn
import torch.nn.functional as F

class AGF(nn.Module):
    """Adaptive Gated Fusion."""
    def __init__(self, embed_dims):
        super().__init__()
        self.gate_convs = nn.ModuleList([
            nn.Conv3d(ed, 1, kernel_size=1) for ed in embed_dims
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        target_size = features[0].shape[2:]  # finest scale
        resized = []
        gates = []

        for feat, conv in zip(features, self.gate_convs):
            feat_up = F.interpolate(feat, size=target_size, mode='trilinear', align_corners=True)
            resized.append(feat_up)
            gate = self.sigmoid(conv(feat_up))
            gates.append(gate)

        fused = torch.sum(torch.stack([g * f for g, f in zip(gates, resized)]), dim=0)
        return fused