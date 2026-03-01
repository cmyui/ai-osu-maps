import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embedding(x: Tensor, d_model: int) -> Tensor:
    """Compute sinusoidal positional embedding for arbitrary scalar values.

    Args:
        x: Scalar values of shape (B,) or (B, 1).
        d_model: Embedding dimension (must be even).

    Returns:
        Embeddings of shape (B, d_model).
    """
    x = x.view(-1, 1)
    half_dim = d_model // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=x.device, dtype=x.dtype)
        / half_dim,
    )
    angles = x * freq.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ScalarEmbedder(nn.Module):
    """Embeds a continuous scalar via sinusoidal encoding + MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: Scalar values of shape (B,).

        Returns:
            Embedding of shape (B, d_model).
        """
        emb = sinusoidal_embedding(x, self.d_model)
        return self.mlp(emb)


class MapperEmbedder(nn.Module):
    """Embeds a mapper identity via learned embedding lookup.

    Index 0 is reserved for null/unknown mapper.
    """

    def __init__(self, num_mappers: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_mappers + 1, d_model)

    def forward(self, mapper_id: Tensor) -> Tensor:
        """Args:
            mapper_id: (B,) long tensor of mapper IDs.

        Returns:
            Embedding of shape (B, d_model).
        """
        return self.emb(mapper_id)
