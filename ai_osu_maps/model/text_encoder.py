from __future__ import annotations

import torch
import torch.nn as nn

TEXT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_ENCODER_DIM = 384
NUM_TEXT_TOKENS = 4


class TextEncoder(nn.Module):
    """Frozen sentence-transformers encoder that produces pseudo-tokens for cross-attention.

    Encodes a text prompt into a 384-dim vector, projects to d_model,
    and expands to NUM_TEXT_TOKENS pseudo-tokens for cross-attention.
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(TEXT_ENCODER_MODEL)
        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(TEXT_ENCODER_DIM, d_model * NUM_TEXT_TOKENS)
        self.d_model = d_model

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Encode text prompts into embeddings.

        Args:
            prompts: List of text strings.

        Returns:
            (B, 384) text embeddings on CPU.
        """
        return torch.from_numpy(self.model.encode(prompts, convert_to_numpy=True))

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project pre-computed text embeddings to cross-attention tokens.

        Args:
            text_embeddings: (B, 384) from encode_text().

        Returns:
            (B, NUM_TEXT_TOKENS, d_model) pseudo-tokens for cross-attention.
        """
        projected = self.projection(text_embeddings)  # (B, d_model * N)
        return projected.view(-1, NUM_TEXT_TOKENS, self.d_model)
