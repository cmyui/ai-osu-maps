import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        window_size: int = 256,
        n_registers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.n_registers = n_registers

        self.registers = nn.Parameter(torch.randn(n_registers, d_model) * 0.02)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build a boolean mask of shape (seq_len, seq_len).

        True means the position is allowed to attend. The sequence is structured
        as [registers..., tokens...]. Every position can attend to all register
        positions. Token positions can additionally attend to tokens within
        window_size // 2 distance.
        """
        total = self.n_registers + seq_len
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)

        # All positions attend to registers (columns 0..n_registers)
        mask[:, : self.n_registers] = True

        # Registers attend to all positions
        mask[: self.n_registers, :] = True

        # Sliding window among token positions
        token_indices = torch.arange(seq_len, device=device)
        row = token_indices.unsqueeze(1)  # (seq_len, 1)
        col = token_indices.unsqueeze(0)  # (1, seq_len)
        half_window = self.window_size // 2
        window_mask = (row - col).abs() <= half_window  # (seq_len, seq_len)

        mask[self.n_registers :, self.n_registers :] = window_mask

        return mask

    def forward(self, x: torch.Tensor, ada_params: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, S, D).
            ada_params: Adaptive parameters of shape (B, D), used to modulate
                the register tokens per-sample.

        Returns:
            Output tensor of shape (B, S, D).
        """
        batch_size, seq_len, _ = x.shape

        # Expand registers per batch and modulate with ada_params
        regs = self.registers.unsqueeze(0).expand(batch_size, -1, -1)
        regs = regs * (1.0 + ada_params.unsqueeze(1))

        # Prepend registers to the sequence
        x_with_regs = torch.cat([regs, x], dim=1)  # (B, R+S, D)

        # QKV projection
        qkv = self.qkv_proj(x_with_regs)  # (B, R+S, 3*D)
        qkv = qkv.reshape(batch_size, -1, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, R+S, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, H, R+S, head_dim)

        # Build and apply sliding window attention mask
        mask = self._build_attention_mask(seq_len, x.device)  # (R+S, R+S)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
        )  # (B, H, R+S, head_dim)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        # Strip register positions from output
        attn_out = attn_out[:, self.n_registers :, :]  # (B, S, D)

        return self.out_proj(attn_out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query tensor of shape (B, S, D).
            context: Context tensor (audio features) of shape (B, T, D).

        Returns:
            Output tensor of shape (B, S, D).
        """
        batch_size = x.shape[0]

        q = self.q_proj(x)  # (B, S, D)
        kv = self.kv_proj(context)  # (B, T, 2*D)

        q = q.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(batch_size, -1, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, T, head_dim)
        k, v = kv.unbind(0)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )  # (B, H, S, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        return self.out_proj(attn_out)
