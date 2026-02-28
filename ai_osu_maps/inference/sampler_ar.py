from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import Tensor

from ai_osu_maps.config import GenerationConfig
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.model.ar_transformer import ARTransformer


def top_k_top_p_filter(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """Filter logits using top-k and/or top-p (nucleus) sampling.

    Args:
        logits: (B, vocab_size)
        top_k: Keep only top k tokens. 0 = disabled.
        top_p: Keep tokens with cumulative prob <= top_p. 1.0 = disabled.

    Returns:
        Filtered logits with -inf for removed tokens.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


@torch.no_grad()
def sample_autoregressively(
    model: ARTransformer,
    tokenizer: Tokenizer,
    audio_features: Tensor,
    difficulty: Tensor,
    cs: Tensor,
    ar: Tensor,
    od: Tensor,
    hp: Tensor,
    config: GenerationConfig,
    audio_mask: Tensor | None = None,
    text_emb: Tensor | None = None,
    device: torch.device | None = None,
    stream: bool = False,
) -> list[int]:
    """Generate token sequence autoregressively.

    Args:
        model: Trained ARTransformer model.
        tokenizer: Tokenizer instance.
        audio_features: (1, T_audio, d_model) audio features.
        difficulty, cs, ar, od, hp: (1,) scalar conditions.
        config: Generation config with temperature, top_k, top_p, etc.
        audio_mask: (1, T_audio) bool mask.
        text_emb: (1, 384) text embeddings or None.
        device: Device for generation.

    Returns:
        List of generated token IDs (excluding SOS, including EOS).
    """
    if device is None:
        device = audio_features.device

    model.eval()
    generated = [tokenizer.sos_id]
    max_tokens = config.max_tokens

    for _ in range(max_tokens):
        # Window the input if it exceeds model's max_seq_len
        input_ids = generated
        if len(input_ids) > model.max_seq_len:
            input_ids = input_ids[-model.max_seq_len :]

        token_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        if config.cfg_scale > 1.0 and text_emb is not None:
            # CFG: conditioned pass
            logits_cond = model.generate_next_token(
                token_tensor, audio_features, difficulty, cs, ar, od, hp,
                audio_mask=audio_mask, text_emb=text_emb,
            )
            # Unconditioned pass (drop text + scalars)
            logits_uncond = model.forward(
                token_tensor, audio_features,
                torch.zeros_like(difficulty), torch.zeros_like(cs),
                torch.zeros_like(ar), torch.zeros_like(od), torch.zeros_like(hp),
                audio_mask=audio_mask, text_emb=None,
                drop_scalars=torch.ones(1, dtype=torch.bool, device=device),
            )[:, -1, :]

            logits = logits_uncond + config.cfg_scale * (logits_cond - logits_uncond)
        else:
            logits = model.generate_next_token(
                token_tensor, audio_features, difficulty, cs, ar, od, hp,
                audio_mask=audio_mask, text_emb=text_emb,
            )

        # Temperature scaling
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Top-k / top-p filtering
        logits = top_k_top_p_filter(logits, config.top_k, config.top_p)

        # Block PAD and SOS tokens
        logits[0, tokenizer.pad_id] = float("-inf")
        logits[0, tokenizer.sos_id] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if stream:
            if next_token in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
                label = {tokenizer.pad_id: "PAD", tokenizer.sos_id: "SOS", tokenizer.eos_id: "EOS"}[next_token]
                sys.stderr.write(f"\r[{len(generated)-1}/{max_tokens}] <{label}>")
            else:
                event = tokenizer.decode(next_token)
                sys.stderr.write(f"\r[{len(generated)-1}/{max_tokens}] {event.type.value}{event.value}")
            sys.stderr.flush()

        if next_token == tokenizer.eos_id:
            break

    if stream:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return generated[1:]  # exclude SOS
