## Replace Transformer backbone with DNN
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, t_dim, h_dim, num_ffns, num_groups, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_groups, embed_dim)
        self.ffn_in = nn.Sequential(
            nn.Linear(t_dim + embed_dim, h_dim, bias=True),
            nn.LeakyReLU()
        )

        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.LeakyReLU()
        ) for _ in range(num_ffns)])

        self.fc = nn.Linear(h_dim, t_dim, bias=True)

    def forward(self, y_imp, group):
        group_emb = self.embed(group)
        z = torch.cat([y_imp, group_emb], dim=-1)
        z = self.ffn_in(z)
        for ffn in self.ffns:
            z = ffn(z)
        z = self.fc(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, t_dim, h_dim, num_ffns):
        super().__init__()
        self.ffn_in = nn.Sequential(
            nn.Linear(2 * t_dim, h_dim, bias=True),
            nn.LeakyReLU()
        )

        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=True),
            nn.LeakyReLU()
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








