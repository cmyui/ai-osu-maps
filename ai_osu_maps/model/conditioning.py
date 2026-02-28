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


class TimestepEmbedder(nn.Module):
    """Embeds flow timestep t in [0, 1] via sinusoidal encoding + MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Args:
            t: Flow timestep of shape (B,).

        Returns:
            Embedding of shape (B, d_model).
        """
        emb = sinusoidal_embedding(t, self.d_model)
        return self.mlp(emb)


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
    """Learnable embedding table for mapper IDs. Index 0 is the null/unknown token."""

    def __init__(self, num_mappers: int, d_model: int) -> None:
        super().__init__()
        # +1 for the null token at index 0
        self.embedding = nn.Embedding(num_mappers + 1, d_model)

    def forward(self, ids: Tensor) -> Tensor:
        """Args:
            ids: Mapper IDs of shape (B,). 0 means null/unknown.

        Returns:
            Embedding of shape (B, d_model).
        """
        return self.embedding(ids)


SCALAR_CONDITION_KEYS = ("difficulty", "cs", "ar", "od", "hp", "year")


class ConditioningModule(nn.Module):
    """Combines all conditioning embedders into a single (B, d_model) vector.

    Each condition is independently droppable via a boolean drop_mask. Dropped
    conditions are replaced with zeros so they contribute nothing to the sum.
    The output vector is intended for AdaLN-Zero modulation in the transformer.
    """

    def __init__(self, d_model: int, num_mappers: int = 4096) -> None:
        super().__init__()
        self.timestep_emb = TimestepEmbedder(d_model)
        self.difficulty_emb = ScalarEmbedder(d_model)
        self.cs_emb = ScalarEmbedder(d_model)
        self.ar_emb = ScalarEmbedder(d_model)
        self.od_emb = ScalarEmbedder(d_model)
        self.hp_emb = ScalarEmbedder(d_model)
        self.year_emb = ScalarEmbedder(d_model)
        self.mapper_emb = MapperEmbedder(num_mappers, d_model)

    def forward(
        self,
        timestep: Tensor,
        difficulty: Tensor,
        cs: Tensor,
        ar: Tensor,
        od: Tensor,
        hp: Tensor,
        mapper_id: Tensor,
        year: Tensor,
        drop_mask: dict[str, Tensor],
    ) -> Tensor:
        """Compute the combined conditioning vector.

        Args:
            timestep: Flow timestep of shape (B,).
            difficulty: Star rating of shape (B,).
            cs: Circle size of shape (B,).
            ar: Approach rate of shape (B,).
            od: Overall difficulty of shape (B,).
            hp: HP drain of shape (B,).
            mapper_id: Mapper IDs of shape (B,), 0 for unknown.
            year: Map year of shape (B,).
            drop_mask: Maps condition name to a boolean tensor of shape (B,).
                       True means drop (replace with zeros).

        Returns:
            Combined conditioning vector of shape (B, d_model).
        """
        embeddings: dict[str, Tensor] = {
            "timestep": self.timestep_emb(timestep),
            "difficulty": self.difficulty_emb(difficulty),
            "cs": self.cs_emb(cs),
            "ar": self.ar_emb(ar),
            "od": self.od_emb(od),
            "hp": self.hp_emb(hp),
            "mapper": self.mapper_emb(mapper_id),
            "year": self.year_emb(year),
        }

        result = torch.zeros_like(embeddings["timestep"])
        for key, emb in embeddings.items():
            mask = drop_mask.get(key)
            if mask is not None:
                # mask shape (B,) -> (B, 1) for broadcasting
                emb = emb * (~mask).unsqueeze(-1).float()
            result = result + emb

        return result
