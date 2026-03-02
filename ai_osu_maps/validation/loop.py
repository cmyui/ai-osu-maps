"""Validation loop: val loss computation and generation-based metrics."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from ai_osu_maps.config import GenerationConfig
from ai_osu_maps.config import TrainingConfig
from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.data.tokenizer import build_token_weight_mask
from ai_osu_maps.inference.sampler import sample_autoregressively
from ai_osu_maps.model.transformer import Transformer
from ai_osu_maps.validation.metrics import aggregate_generation_metrics

if TYPE_CHECKING:
    from ai_osu_maps.data.dataset import BeatmapDataset

logger = logging.getLogger(__name__)

# NLL category definitions
_TIMING_TYPES = (
    EventType.SNAPPING,
    EventType.BEAT,
    EventType.MEASURE,
    EventType.TIMING_POINT,
)
_OBJECT_TYPES = (
    EventType.CIRCLE,
    EventType.SLIDER_HEAD,
    EventType.SPINNER,
    EventType.SLIDER_END,
)
_POSITION_TYPES = (
    EventType.POS,
    EventType.DISTANCE,
)


def _build_category_mask(
    token_ids: torch.Tensor,
    tokenizer: Tokenizer,
    event_types: tuple[EventType, ...],
) -> torch.Tensor:
    """Build a boolean mask selecting tokens belonging to given event types."""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for et in event_types:
        start, end = tokenizer.event_type_range(et)
        mask |= (token_ids >= start) & (token_ids <= end)
    return mask


@torch.no_grad()
def compute_val_loss(
    model: Transformer,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    tokenizer: Tokenizer,
    training_config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Compute validation loss over the entire val set.

    Conditioning dropout is disabled (drop_mask=None).

    Returns dict with val/loss, val/loss_unweighted, val/count_loss,
    val/perplexity, and per-category NLL keys.
    """
    model.eval()

    total_weighted_loss = 0.0
    total_unweighted_loss = 0.0
    total_count_loss = 0.0
    total_tokens = 0
    total_batches = 0

    # Per-category accumulators
    cat_loss: dict[str, float] = {"timing": 0.0, "objects": 0.0, "position": 0.0}
    cat_count: dict[str, int] = {"timing": 0, "objects": 0, "position": 0}

    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for batch in val_loader:
        token_ids = batch["token_ids"].to(device)
        token_mask = batch["token_mask"].to(device)
        audio_features = batch["audio_features"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        difficulty = batch["difficulty"].to(device)
        cs = batch["cs"].to(device)
        ar = batch["ar"].to(device)
        od = batch["od"].to(device)
        hp = batch["hp"].to(device)
        mapper_id = batch["mapper_id"].to(device)
        year = batch["year"].to(device)
        num_objects = batch["num_objects"].to(device)

        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        target_mask = token_mask[:, 1:]

        with autocast(
            device.type,
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            logits, log_count_pred = model(
                input_ids,
                audio_features,
                difficulty,
                cs,
                ar,
                od,
                hp,
                mapper_id,
                year,
                audio_mask=audio_mask,
                text_emb=None,
                drop_mask=None,
                predict_count=True,
            )

            loss_per_token = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=tokenizer.pad_id,
                reduction="none",
            ).reshape(target_ids.shape)

        # Weighted loss (same as training)
        weights = build_token_weight_mask(
            target_ids,
            tokenizer,
            training_config.rhythm_weight,
            training_config.object_weight,
            training_config.position_weight,
        )
        masked_weighted = (loss_per_token * weights * target_mask).sum()
        masked_unweighted = (loss_per_token * target_mask).sum()
        n_tokens = target_mask.sum().item()

        total_weighted_loss += masked_weighted.item()
        total_unweighted_loss += masked_unweighted.item()
        total_tokens += n_tokens
        total_batches += 1

        # Count loss
        log_target = torch.log(num_objects.float().clamp(min=1))
        count_loss = nn.functional.l1_loss(log_count_pred, log_target)
        total_count_loss += count_loss.item()

        # Per-category NLL
        for cat_name, event_types in [
            ("timing", _TIMING_TYPES),
            ("objects", _OBJECT_TYPES),
            ("position", _POSITION_TYPES),
        ]:
            cat_mask = _build_category_mask(target_ids, tokenizer, event_types)
            valid_mask = cat_mask & target_mask.bool()
            n_cat = valid_mask.sum().item()
            if n_cat > 0:
                cat_loss[cat_name] += (loss_per_token * valid_mask).sum().item()
                cat_count[cat_name] += n_cat

    # Aggregate
    n = max(total_tokens, 1)
    avg_weighted = total_weighted_loss / n
    avg_unweighted = total_unweighted_loss / n
    avg_count_loss = total_count_loss / max(total_batches, 1)
    perplexity = math.exp(min(avg_unweighted, 20.0))  # clamp to avoid overflow

    metrics: dict[str, float] = {
        "val/loss": avg_weighted,
        "val/loss_unweighted": avg_unweighted,
        "val/count_loss": avg_count_loss,
        "val/perplexity": perplexity,
    }

    for cat_name in ("timing", "objects", "position"):
        nll = cat_loss[cat_name] / max(cat_count[cat_name], 1)
        metrics[f"val/nll_{cat_name}"] = nll

    return metrics


@torch.no_grad()
def run_validation(
    ema_model: Transformer,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    val_dataset: BeatmapDataset,
    tokenizer: Tokenizer,
    training_config: TrainingConfig,
    device: torch.device,
    *,
    n_generate: int = 4,
    max_generate_tokens: int = 512,
) -> dict[str, float]:
    """Run full validation: loss computation + generation metrics.

    Args:
        ema_model: EMA model weights (unwrapped, not DDP).
        val_loader: Validation DataLoader.
        val_dataset: Validation dataset (for sampling individual items).
        tokenizer: Tokenizer instance.
        training_config: Training config (for loss weights).
        device: Torch device.
        n_generate: Number of samples to generate for quality metrics.
        max_generate_tokens: Max tokens per generation (keep small for speed).

    Returns:
        Combined metrics dict with val/ and gen/ prefixed keys.
    """
    logger.info("Running validation...")

    # Phase 1: Val loss
    metrics = compute_val_loss(
        ema_model,
        val_loader,
        tokenizer,
        training_config,
        device,
    )
    logger.info(
        "Val loss=%.4f unweighted=%.4f perplexity=%.2f count_loss=%.4f",
        metrics["val/loss"],
        metrics["val/loss_unweighted"],
        metrics["val/perplexity"],
        metrics["val/count_loss"],
    )
    logger.info(
        "Val NLL timing=%.4f objects=%.4f position=%.4f",
        metrics["val/nll_timing"],
        metrics["val/nll_objects"],
        metrics["val/nll_position"],
    )

    # Phase 2: Generation metrics
    n_samples = min(n_generate, len(val_dataset))
    if n_samples == 0:
        return metrics

    gen_config = GenerationConfig(
        temperature=1.0,
        timing_temperature=1.0,
        top_k=1,  # greedy decode
        top_p=1.0,
        max_tokens=max_generate_tokens,
        monotonic_time=True,
    )

    generated_samples: list[list[int]] = []
    ground_truth_samples: list[list[int]] = []

    ema_model.eval()

    # Sample evenly spaced indices from the val set for diversity
    indices = [i * len(val_dataset) // n_samples for i in range(n_samples)]

    for idx in indices:
        sample = val_dataset[idx]

        audio_features = sample["audio_features"].unsqueeze(0).to(device)
        difficulty = sample["difficulty"].unsqueeze(0).to(device)
        cs_val = sample["cs"].unsqueeze(0).to(device)
        ar_val = sample["ar"].unsqueeze(0).to(device)
        od_val = sample["od"].unsqueeze(0).to(device)
        hp_val = sample["hp"].unsqueeze(0).to(device)
        mapper_id = sample["mapper_id"].unsqueeze(0).to(device)
        year = sample["year"].unsqueeze(0).to(device)

        gen_tokens = sample_autoregressively(
            ema_model,
            tokenizer,
            audio_features,
            difficulty,
            cs_val,
            ar_val,
            od_val,
            hp_val,
            mapper_id,
            year,
            gen_config,
            device=device,
        )
        generated_samples.append(gen_tokens)

        gt_tokens = sample["token_ids"].tolist()
        # Strip SOS/PAD from ground truth for fair comparison
        gt_tokens = [
            t for t in gt_tokens if t not in (tokenizer.pad_id, tokenizer.sos_id)
        ]
        ground_truth_samples.append(gt_tokens)

    gen_metrics = aggregate_generation_metrics(
        generated_samples,
        ground_truth_samples,
        tokenizer,
    )
    metrics.update(gen_metrics)

    logger.info(
        "Gen: seq_len=%.0f objects=%.0f obj_ratio=%.2f "
        "monotonic=%.0f%% eos=%.0f%% js_div=%.4f",
        gen_metrics.get("gen/seq_length_mean", 0),
        gen_metrics.get("gen/object_count_mean", 0),
        gen_metrics.get("gen/object_count_ratio", 0),
        gen_metrics.get("gen/monotonic_time_frac", 0) * 100,
        gen_metrics.get("gen/eos_frac", 0) * 100,
        gen_metrics.get("gen/js_divergence", 0),
    )

    return metrics
