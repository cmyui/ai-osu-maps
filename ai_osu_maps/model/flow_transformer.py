import torch
import torch.nn as nn
from torch import Tensor

from ai_osu_maps.config import ModelConfig
from ai_osu_maps.model.attention import CrossAttention
from ai_osu_maps.model.attention import SlidingWindowSelfAttention
from ai_osu_maps.model.conditioning import ConditioningModule


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero modulation.

    Projects a conditioning vector to 3 modulation parameters per sublayer:
    (scale, shift, gate) used for pre-norm modulation and residual gating.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )
        # Initialize to zero so residual paths start as identity
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute 3 modulation parameters from a conditioning vector.

        Args:
            cond: Conditioning vector of shape (B, D).

        Returns:
            Three tensors of shape (B, 1, D): scale, shift, gate.
        """
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 3*D)
        return params.chunk(3, dim=-1)


class FlowTransformerBlock(nn.Module):
    """Single transformer block with AdaLN-Zero modulation.

    Pattern: AdaLN -> self-attn -> residual -> AdaLN -> cross-attn -> residual
             -> AdaLN -> FFN -> residual.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        window_size: int,
        n_registers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SlidingWindowSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            n_registers=n_registers,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.adaln_self = AdaLNZero(d_model)
        self.adaln_cross = AdaLNZero(d_model)
        self.adaln_ffn = AdaLNZero(d_model)

    def forward(
        self, x: Tensor, audio_features: Tensor, cond_emb: Tensor
    ) -> Tensor:
        # Self-attention with AdaLN-Zero
        scale, shift, gate = self.adaln_self(cond_emb)
        h = self.norm1(x) * (1.0 + scale) + shift
        h = self.self_attn(h, cond_emb)
        x = x + gate * h

        # Cross-attention with AdaLN-Zero
        scale, shift, gate = self.adaln_cross(cond_emb)
        h = self.norm2(x) * (1.0 + scale) + shift
        h = self.cross_attn(h, audio_features)
        x = x + gate * h

        # FFN with AdaLN-Zero
        scale, shift, gate = self.adaln_ffn(cond_emb)
        h = self.norm3(x) * (1.0 + scale) + shift
        h = self.ffn(h)
        x = x + gate * h

        return x


class ObjectCountPredictor(nn.Module):
    """Predicts log(num_objects) from mean-pooled audio features."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, audio_features: Tensor) -> Tensor:
        """Predict log(num_objects) from audio features.

        Args:
            audio_features: Audio features of shape (B, T, D).

        Returns:
            Log object count of shape (B,).
        """
        pooled = audio_features.mean(dim=1)  # (B, D)
        return self.mlp(pooled).squeeze(-1)  # (B,)


class FlowTransformer(nn.Module):
    """Flow-matching transformer for osu! beatmap generation.

    Takes noised object representations and conditioning signals, produces
    velocity predictions for the flow ODE. AudioEncoder is kept external
    for memory efficiency.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model

        self.input_proj = nn.Linear(config.obj_dim, d)
        self.conditioning = ConditioningModule(d)

        self.blocks = nn.ModuleList([
            FlowTransformerBlock(
                d_model=d,
                n_heads=config.n_heads,
                window_size=config.window_size,
                n_registers=config.n_registers,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.output_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, config.obj_dim)

        self.object_count_predictor = ObjectCountPredictor(d)

    def forward(
        self,
        x_t: Tensor,
        timestep: Tensor,
        audio_features: Tensor,
        difficulty: Tensor,
        cs: Tensor,
        ar: Tensor,
        od: Tensor,
        hp: Tensor,
        mapper_id: Tensor,
        year: Tensor,
        drop_mask: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x_t: Noised object representations of shape (B, S, obj_dim).
            timestep: Flow timestep of shape (B,).
            audio_features: Encoded audio of shape (B, T, d_model).
            difficulty: Star rating of shape (B,).
            cs: Circle size of shape (B,).
            ar: Approach rate of shape (B,).
            od: Overall difficulty of shape (B,).
            hp: HP drain of shape (B,).
            mapper_id: Mapper IDs of shape (B,).
            year: Map year of shape (B,).
            drop_mask: Maps condition name to boolean tensor of shape (B,).

        Returns:
            velocity: Predicted flow velocity of shape (B, S, obj_dim).
            log_num_objects: Predicted log object count of shape (B,).
        """
        cond_emb = self.conditioning(
            timestep, difficulty, cs, ar, od, hp, mapper_id, year, drop_mask,
        )

        x = self.input_proj(x_t)  # (B, S, d_model)

        for block in self.blocks:
            x = block(x, audio_features, cond_emb)

        velocity = self.output_proj(self.output_norm(x))  # (B, S, obj_dim)

        log_num_objects = self.object_count_predictor(audio_features)

        return velocity, log_num_objects
