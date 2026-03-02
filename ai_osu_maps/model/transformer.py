from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ai_osu_maps.model.conditioning import MapperEmbedder
from ai_osu_maps.model.conditioning import ScalarEmbedder


class RoPE(nn.Module):
    """Rotary Positional Encoding."""

    def __init__(self, d_head: int, max_len: int = 8192) -> None:
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_len, d_head//2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply RoPE to x of shape (..., seq_len, d_head)."""
        seq_len = x.shape[-2]
        cos = self.cos[offset : offset + seq_len]  # (S, d_head//2)
        sin = self.sin[offset : offset + seq_len]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)


class AdaLN(nn.Module):
    """Adaptive Layer Norm with scale and shift from conditioning."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (normed_x * (1 + scale) + shift, gate).

        Args:
            x: (B, S, d_model)
            cond: (B, d_model)

        Returns:
            modulated: (B, S, d_model)
            gate: (B, 1, d_model)
        """
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 3*d)
        scale, shift, gate = params.chunk(3, dim=-1)
        modulated = self.norm(x) * (1 + scale) + shift
        return modulated, gate


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor, rope: RoPE, offset: int = 0) -> Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, D)
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = rope(q, offset)
        k = rope(k, offset)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (B, S, d_model) queries
            context: (B, T, d_model) keys/values (audio + text concatenated)
            context_mask: (B, T) bool mask, True = attend, False = ignore
        """
        B, S, _ = x.shape
        T = context.shape[1]

        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, T, 2, self.n_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            # (B, T) -> (B, 1, 1, T) for broadcasting over heads and queries
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with optional cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        *,
        has_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.has_cross_attention = has_cross_attention

        self.adaln_self = AdaLN(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)

        if has_cross_attention:
            self.adaln_cross = AdaLN(d_model)
            self.cross_attn = CrossAttention(d_model, n_heads, dropout)

        self.adaln_ff = AdaLN(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        rope: RoPE,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        offset: int = 0,
    ) -> Tensor:
        # Self-attention with AdaLN
        normed, gate = self.adaln_self(x, cond)
        x = x + gate * self.self_attn(normed, rope, offset)

        # Cross-attention (every 2nd layer)
        if self.has_cross_attention and context is not None:
            normed, gate = self.adaln_cross(x, cond)
            x = x + gate * self.cross_attn(normed, context, context_mask)

        # Feed-forward
        normed, gate = self.adaln_ff(x, cond)
        x = x + gate * self.ff(normed)

        return x


class Transformer(nn.Module):
    """Decoder-only autoregressive transformer for beatmap generation.

    Conditioned on audio (MERT features) and text (sentence-transformer)
    via cross-attention, and scalar conditions (difficulty, CS, AR, OD, HP)
    via AdaLN modulation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        mert_dim: int = 768,
        text_dim: int = 384,
        n_text_tokens: int = 4,
        num_mappers: int = 4096,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_text_tokens = n_text_tokens

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # RoPE
        d_head = d_model // n_heads
        self.rope = RoPE(d_head, max_len=max_seq_len + 512)

        # Audio projection (MERT features already projected to d_model by AudioEncoder)
        # No extra projection needed if audio_encoder already outputs d_model

        # Text projection
        self.text_proj = nn.Linear(text_dim, d_model * n_text_tokens)

        # Scalar conditioning (AdaLN)
        self.scalar_embeddings: dict[str, ScalarEmbedder] = {}
        self.difficulty_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["difficulty"] = self.difficulty_emb
        self.cs_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["cs"] = self.cs_emb
        self.ar_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["ar"] = self.ar_emb
        self.od_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["od"] = self.od_emb
        self.hp_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["hp"] = self.hp_emb
        self.year_emb = ScalarEmbedder(d_model)
        self.scalar_embeddings["year"] = self.year_emb

        # Mapper conditioning
        self.mapper_emb = MapperEmbedder(num_mappers, d_model)

        # Object count predictor (auxiliary loss head)
        self.count_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Transformer blocks: cross-attention every 2nd layer
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    dropout,
                    has_cross_attention=(i % 2 == 1),
                )
                for i in range(n_layers)
            ],
        )

        # Output head
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _build_conditioning(
        self,
        difficulty: Tensor,
        cs: Tensor,
        ar: Tensor,
        od: Tensor,
        hp: Tensor,
        mapper_id: Tensor,
        year: Tensor,
        drop_mask: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Sum scalar + mapper condition embeddings into (B, d_model).

        Args:
            drop_mask: Per-condition (B,) bool tensors; True = drop that condition.
        """
        scalars = {
            "difficulty": difficulty,
            "cs": cs,
            "ar": ar,
            "od": od,
            "hp": hp,
            "year": year,
        }
        cond = torch.zeros(
            difficulty.shape[0],
            self.d_model,
            device=difficulty.device,
        )
        for key, value in scalars.items():
            emb = self.scalar_embeddings[key](value)
            if drop_mask is not None and key in drop_mask:
                emb = emb * (~drop_mask[key]).unsqueeze(-1).float()
            cond = cond + emb

        mapper_out = self.mapper_emb(mapper_id)
        if drop_mask is not None and "mapper" in drop_mask:
            mapper_out = mapper_out * (~drop_mask["mapper"]).unsqueeze(-1).float()
        cond = cond + mapper_out

        return cond

    def _build_context(
        self,
        audio_features: Tensor,
        audio_mask: Tensor | None,
        text_emb: Tensor | None,
        drop_text: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Build cross-attention context from audio + text.

        Returns:
            context: (B, T_audio + N_text, d_model)
            context_mask: (B, T_audio + N_text) bool or None
        """
        B = audio_features.shape[0]
        context = audio_features  # (B, T_audio, d_model)
        mask = audio_mask  # (B, T_audio) or None

        if text_emb is not None:
            text_tokens = self.text_proj(text_emb)  # (B, d_model * N)
            text_tokens = text_tokens.view(B, self.n_text_tokens, self.d_model)

            if drop_text is not None:
                text_tokens = (
                    text_tokens * (~drop_text).unsqueeze(-1).unsqueeze(-1).float()
                )

            context = torch.cat([context, text_tokens], dim=1)
            if mask is not None:
                text_mask = torch.ones(
                    B,
                    self.n_text_tokens,
                    dtype=torch.bool,
                    device=mask.device,
                )
                mask = torch.cat([mask, text_mask], dim=1)

        return context, mask

    def forward(
        self,
        token_ids: Tensor,
        audio_features: Tensor,
        difficulty: Tensor,
        cs: Tensor,
        ar: Tensor,
        od: Tensor,
        hp: Tensor,
        mapper_id: Tensor,
        year: Tensor,
        audio_mask: Tensor | None = None,
        text_emb: Tensor | None = None,
        drop_mask: dict[str, Tensor] | None = None,
        drop_text: Tensor | None = None,
        predict_count: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass for teacher forcing.

        Args:
            token_ids: (B, S) input token IDs
            audio_features: (B, T_audio, d_model) pre-computed MERT features
            difficulty, cs, ar, od, hp: (B,) scalar conditions
            mapper_id: (B,) long mapper identity
            year: (B,) float year condition
            audio_mask: (B, T_audio) bool mask for audio padding
            text_emb: (B, 384) text embeddings or None
            drop_mask: Per-condition (B,) bool tensors for CFG dropout
            drop_text: (B,) bool - drop text condition for CFG
            predict_count: If True, also return log(object_count) prediction.

        Returns:
            logits (B, S, vocab_size) when predict_count=False, else
            (logits, log_count_pred (B,)) tuple.
        """
        # Token embeddings
        x = self.token_emb(token_ids) * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        # Conditioning
        cond = self._build_conditioning(
            difficulty,
            cs,
            ar,
            od,
            hp,
            mapper_id,
            year,
            drop_mask,
        )
        context, context_mask = self._build_context(
            audio_features,
            audio_mask,
            text_emb,
            drop_text,
        )

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond, self.rope, context, context_mask)

        # Count prediction (before ln_final, from raw hidden states)
        log_count_pred: Tensor | None = None
        if predict_count:
            pooled = x.mean(dim=1)  # (B, d_model)
            log_count_pred = self.count_predictor(pooled).squeeze(-1)  # (B,)

        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if log_count_pred is not None:
            return logits, log_count_pred
        return logits

    @torch.no_grad()
    def generate_next_token(
        self,
        token_ids: Tensor,
        audio_features: Tensor,
        difficulty: Tensor,
        cs: Tensor,
        ar: Tensor,
        od: Tensor,
        hp: Tensor,
        mapper_id: Tensor,
        year: Tensor,
        audio_mask: Tensor | None = None,
        text_emb: Tensor | None = None,
        drop_mask: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Get logits for the next token (last position only).

        Args:
            token_ids: (B, S) current token sequence
            Others: same as forward()

        Returns:
            logits: (B, vocab_size) for the next token
        """
        logits = self.forward(
            token_ids,
            audio_features,
            difficulty,
            cs,
            ar,
            od,
            hp,
            mapper_id,
            year,
            audio_mask=audio_mask,
            text_emb=text_emb,
            drop_mask=drop_mask,
        )
        return logits[:, -1, :]  # (B, vocab_size)
