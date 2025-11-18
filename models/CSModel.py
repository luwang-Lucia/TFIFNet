import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class CSModel(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        ).to(self.device)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, x1, x2):
        channel_mask = self.global_pool(x1)
        channel_mask = self.channel_interaction(channel_mask)
        m1 = x1 * channel_mask
        spatial_mask = self.spatial_interaction(m1)
        m2 = m1 * spatial_mask
        output = x2 * m2
        return output