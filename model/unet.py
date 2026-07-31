import torch.nn as nn
import torch.nn.functional as F

from .embeddings import SinusoidalPositionEmbeddings
from .blocks import DoubleConv
from .attention import AttentionBlock

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, time_dim=256):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.down1 = DoubleConv(in_channels, 64, time_dim)
        self.down2 = DoubleConv(64, 128, time_dim)
        self.down3 = DoubleConv(128, 256, time_dim)

        self.attn = AttentionBlock(256)

        self.pool = nn.MaxPool2d(2)

        self.up1 = DoubleConv(256 + 128, 128, time_dim)
        self.up2 = DoubleConv(128 + 64, 64, time_dim)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):

        t = self.time_embedding(t)

        d1 = self.down1(x, t)
        d2 = self.down2(self.pool(d1), t)
        d3 = self.down3(self.pool(d2), t)

        d3 = self.attn(d3)

        up1 = F.interpolate(d3, scale_factor=2)
        up1 = self.up1(torch.cat([up1, d2], dim=1), t)

        up2 = F.interpolate(up1, scale_factor=2)
        up2 = self.up2(torch.cat([up2, d1], dim=1), t)

        return self.final(up2)
