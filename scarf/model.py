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

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

        self.scale = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn_output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class TransformerLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = PositionwiseFeedforward(emb_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.dropout1(self.norm1(x + attn_output))

        ff_output = self.feedforward(x)
        x = self.dropout2(self.norm2(x + ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        num_heads,
        hidden_dim,
        num_layers,
        features_low,
        features_high,
        corruption_rate=0.6,
        dropout=0.0,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, emb_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, emb_dim))  # Adjust the sequence length

        self.layers = nn.ModuleList([
            TransformerLayer(emb_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

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
        x = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        x_corrupted = self.corrupt_input(x)

        for layer in self.layers:
            x = layer(x)

        embeddings = x[:, : x.size(1) // 2, :]  # Consider only first half for embeddings
        embeddings_corrupted = x_corrupted[:, : x_corrupted.size(1) // 2, :]  # Same for corrupted

        return embeddings, embeddings_corrupted

