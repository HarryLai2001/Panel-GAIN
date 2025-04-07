import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, t_dim, h_dim):
        super().__init__()
        pe = torch.zeros(t_dim, h_dim)
        pos = torch.arange(0, t_dim, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, h_dim, 2).float() * (-math.log(10000.0) / h_dim))
        pe[:, 0::2] = torch.sin(pos[:, None] * div_term[None, :])
        pe[:, 1::2] = torch.cos(pos[:, None] * div_term[None, :])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.scale = in_dim ** -0.5

        self.layer_norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # LayerNorm
        x = self.layer_norm(x)
        # Attention
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        w = q @ k.transpose(-1, -2)
        w = self.softmax(w * self.scale) @ v
        # FFN
        w = self.fc(w)
        w = self.relu(w)
        return w



class Generator(nn.Module):
    def __init__(self, t_dim, h_dims, num_groups, embed_dim):
        super().__init__()
        self.t_dim = t_dim

        self.embed = nn.Embedding(num_groups, embed_dim)
        self.proj_in = nn.Linear(1 + embed_dim, h_dims[0], bias=True)
        self.pe = PositionalEncoding(t_dim, h_dims[0])

        self.encoders = nn.ModuleList()
        for i in range(len(h_dims) - 1):
            self.encoders.append(Encoder(h_dims[i], h_dims[i+1]))
        self.encoders.append(Encoder(h_dims[-1], 1))
        self.fc = nn.Linear(t_dim, t_dim, bias=True)

    def forward(self, y_imp, group):
        group_emb = self.embed(group).unsqueeze(1).repeat(1, self.t_dim, 1)
        z = torch.cat([y_imp.unsqueeze(-1), group_emb], dim=-1)
        z = self.proj_in(z)
        z = self.pe(z)
        for encoder in self.encoders:
            z = encoder(z)
        z = z.squeeze(-1)
        z =  self.fc(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, t_dim, h_dims):
        super().__init__()
        self.proj_in = nn.Linear(2, h_dims[0], bias=True)
        self.pe = PositionalEncoding(t_dim, h_dims[0])

        self.encoders = nn.ModuleList()
        for i in range(len(h_dims) - 1):
            self.encoders.append(Encoder(h_dims[i], h_dims[i + 1]))
        self.encoders.append(Encoder(h_dims[-1], 1))
        self.fc = nn.Linear(t_dim, t_dim, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, y0_com, h):
        z = torch.cat([y0_com.unsqueeze(-1), h.unsqueeze(-1)], dim=-1)
        z = self.proj_in(z)
        z = self.pe(z)
        for encoder in self.encoders:
            z = encoder(z)
        z = z.squeeze(-1)
        z = self.fc(z)
        return self.sig(z)


