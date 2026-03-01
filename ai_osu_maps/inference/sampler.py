from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import Tensor

from ai_osu_maps.config import GenerationConfig
from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.tokenizer import MILLISECONDS_PER_STEP, Tokenizer
from ai_osu_maps.model.transformer import Transformer


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


def build_temperature_tensor(
    tokenizer: Tokenizer,
    base_temperature: float,
    timing_temperature: float,
    device: torch.device,
) -> Tensor:
    """Build per-token temperature vector.

    Timing-structural tokens (BEAT, MEASURE, TIMING_POINT) get a low temperature
    so the model commits to them near-deterministically.
    TIME_SHIFT and SNAPPING get a moderate temperature (midpoint).
    Everything else uses the base temperature.

    Returns:
        (vocab_size,) temperature tensor.
    """
    temps = torch.full((tokenizer.vocab_size,), base_temperature, device=device)

    # Structural timing tokens — near-greedy
    for event_type in (EventType.BEAT, EventType.MEASURE, EventType.TIMING_POINT):
        start, end = tokenizer.event_type_range(event_type)
        temps[start : end + 1] = timing_temperature

    # TIME_SHIFT and SNAPPING — midpoint between timing and base
    mid_temperature = (base_temperature + timing_temperature) / 2
    for event_type in (EventType.TIME_SHIFT, EventType.SNAPPING):
        start, end = tokenizer.event_type_range(event_type)
        temps[start : end + 1] = mid_temperature

    return temps


def apply_monotonic_time_constraint(
    logits: Tensor,
    tokenizer: Tokenizer,
    cumulative_time_steps: int,
) -> Tensor:
    """Mask out TIME_SHIFT values that would make cumulative time go negative.

    Args:
        logits: (1, vocab_size) raw logits.
        tokenizer: Tokenizer instance.
        cumulative_time_steps: Current cumulative time in 10ms steps.

    Returns:
        logits with backward-going TIME_SHIFT tokens masked to -inf.
    """
    ts_start, ts_end = tokenizer.event_type_range(EventType.TIME_SHIFT)
    er = tokenizer.event_range[EventType.TIME_SHIFT]

    for token_id in range(ts_start, ts_end + 1):
        delta = er.min_value + (token_id - ts_start)
        if cumulative_time_steps + delta < 0:
            logits[0, token_id] = float("-inf")

    return logits


@torch.no_grad()
def sample_autoregressively(
    model: Transformer,
    tokenizer: Tokenizer,
    audio_features: Tensor,
    difficulty: Tensor,
    cs: Tensor,
    ar: Tensor,
    od: Tensor,
    hp: Tensor,
    mapper_id: Tensor,
    year: Tensor,
    config: GenerationConfig,
    audio_mask: Tensor | None = None,
    text_emb: Tensor | None = None,
    device: torch.device | None = None,
    stream: bool = False,
) -> list[int]:
    """Generate token sequence autoregressively.

    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        audio_features: (1, T_audio, d_model) audio features.
        difficulty, cs, ar, od, hp: (1,) scalar conditions.
        mapper_id: (1,) long mapper identity.
        year: (1,) float year condition.
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

    # Per-token-type temperature
    temp_tensor = build_temperature_tensor(
        tokenizer, config.temperature, config.timing_temperature, device,
    )

    # Monotonic time tracking
    cumulative_time_steps = 0
    ts_start, ts_end = tokenizer.event_type_range(EventType.TIME_SHIFT)
    ts_er = tokenizer.event_range[EventType.TIME_SHIFT]

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
                mapper_id, year,
                audio_mask=audio_mask, text_emb=text_emb,
            )
            # Unconditioned pass (drop all conditions)
            uncond_drop_mask = {
                key: torch.ones(1, dtype=torch.bool, device=device)
                for key in ("difficulty", "cs", "ar", "od", "hp", "mapper", "year")
            }
            logits_uncond = model.forward(
                token_tensor, audio_features,
                torch.zeros_like(difficulty), torch.zeros_like(cs),
                torch.zeros_like(ar), torch.zeros_like(od), torch.zeros_like(hp),
                mapper_id, year,
                audio_mask=audio_mask, text_emb=None,
                drop_mask=uncond_drop_mask,
            )[:, -1, :]

            logits = logits_uncond + config.cfg_scale * (logits_cond - logits_uncond)
        else:
            logits = model.generate_next_token(
                token_tensor, audio_features, difficulty, cs, ar, od, hp,
                mapper_id, year,
                audio_mask=audio_mask, text_emb=text_emb,
            )

        # Per-token-type temperature scaling
        logits = logits / temp_tensor.unsqueeze(0)

        # Monotonic time constraint
        if config.monotonic_time:
            logits = apply_monotonic_time_constraint(
                logits, tokenizer, cumulative_time_steps,
            )

        # Top-k / top-p filtering
        logits = top_k_top_p_filter(logits, config.top_k, config.top_p)

        # Block PAD and SOS tokens
        logits[0, tokenizer.pad_id] = float("-inf")
        logits[0, tokenizer.sos_id] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Track cumulative time
        if ts_start <= next_token <= ts_end:
            delta = ts_er.min_value + (next_token - ts_start)
            cumulative_time_steps += delta

        generated.append(next_token)

        if stream:
            time_ms = cumulative_time_steps * MILLISECONDS_PER_STEP
            if next_token in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
                label = {tokenizer.pad_id: "PAD", tokenizer.sos_id: "SOS", tokenizer.eos_id: "EOS"}[next_token]
                sys.stderr.write(f"\r[{len(generated)-1}/{max_tokens} t={time_ms}ms] <{label}>")
            else:
                event = tokenizer.decode(next_token)
                sys.stderr.write(f"\r[{len(generated)-1}/{max_tokens} t={time_ms}ms] {event.type.value}{event.value}")
            sys.stderr.flush()

        if next_token == tokenizer.eos_id:
            break

    if stream:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return generated[1:]  # exclude SOS
