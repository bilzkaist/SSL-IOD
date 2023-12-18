import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dropout=0.0):
        super().__init__()
        self.conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


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

class SelfAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(emb_dim, dtype=torch.float32)))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output


class SSLNet(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        features_low,
        features_high,
        num_hidden=4,
        head_depth=2,
        cnn_channels=16,
        cnn_kernel_size=3,
        cnn_stride=1,
        cnn_padding=1,
        corruption_rate=0.6,
        dropout=0.0,
    ):
        super().__init__()

        self.cnn = CNNBlock(1, cnn_channels, cnn_kernel_size, stride=cnn_stride, padding=cnn_padding, dropout=dropout)
        self.attention = SelfAttention(emb_dim)
        self.encoder = MLP(input_dim, emb_dim, num_hidden, dropout)
        self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

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
        embeddings = self.encoder(x)
        attn_output = self.attention(embeddings)
        embeddings_with_attention = embeddings + attn_output  # Combine attention output with embeddings
        embeddings_processed = self.pretraining_head(embeddings_with_attention)

        embeddings_corrupted = self.encoder(x_corrupted)
        attn_output_corrupted = self.attention(embeddings_corrupted)
        embeddings_with_attention_corrupted = embeddings_corrupted + attn_output_corrupted
        embeddings_processed_corrupted = self.pretraining_head(embeddings_with_attention_corrupted)

        return embeddings_processed, embeddings_processed_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)
