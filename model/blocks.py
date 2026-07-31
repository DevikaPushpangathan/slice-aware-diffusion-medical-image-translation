import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_dim, out_c)

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )

    def forward(self, x, t):
        h = self.conv(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        return h + time_emb
