## Replication code for GAIN
## Adapted from https://github.com/jsyoon0823/GAIN
## Originally written in tensorflow, and we rewrite the codes in pytorch
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, t_dim, h_dim, num_ffns):
        super().__init__()
        self.ffn_in = nn.Sequential(
            nn.Linear(2 * t_dim, h_dim, bias=True),
            nn.ReLU()
        )

        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.ReLU()
        ) for _ in range(num_ffns)])

        self.ffn_out = nn.Sequential(
            nn.Linear(h_dim, t_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, y_imp, m):
        z = torch.cat([y_imp, m], dim=-1)
        z = self.ffn_in(z)
        for ffn in self.ffns:
            z = ffn(z)
        z = self.ffn_out(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, t_dim, h_dim, num_ffns):
        super().__init__()
        self.ffn_in = nn.Sequential(
            nn.Linear(2 * t_dim, h_dim, bias=True),
            nn.ReLU()
        )

        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.ReLU()
        ) for _ in range(num_ffns)])

        self.logit = nn.Linear(h_dim, t_dim, bias=True)
        self.prob = nn.Sigmoid()

    def forward(self, y0_com, h):
        z = torch.cat([y0_com, h], dim=-1)
        z = self.ffn_in(z)
        for ffn in self.ffns:
            z = ffn(z)
        z = self.logit(z)
        return self.prob(z)








