import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        num_hidden=4,
        head_depth=2,
        corruption_rate=0.6,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth, dropout)

        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_len = int(corruption_rate * input_dim)

    def corrupt_input(self, x):
        batch_size, m = x.size()
        corruption_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        x_random = self.marginals.sample((batch_size,)).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)
        return x_corrupted

    def forward(self, x):
        x_corrupted = self.corrupt_input(x)
        embeddings = self.pretraining_head(self.encoder(x))
        embeddings_corrupted = self.pretraining_head(self.encoder(x_corrupted))
        return embeddings, embeddings_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
