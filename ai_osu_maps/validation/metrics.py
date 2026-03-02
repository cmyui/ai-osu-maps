"""Pure functions for computing generation quality metrics on token ID lists."""

from __future__ import annotations

import math
from collections import Counter

from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.tokenizer import MILLISECONDS_PER_STEP
from ai_osu_maps.data.tokenizer import Tokenizer

OBJECT_EVENT_TYPES = frozenset(
    {EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.SPINNER},
)

TIMING_EVENT_TYPES = frozenset(
    {
        EventType.TIME_SHIFT,
        EventType.SNAPPING,
        EventType.BEAT,
        EventType.MEASURE,
        EventType.TIMING_POINT,
    },
)

ALL_EVENT_TYPES = frozenset(et for et in EventType)


def compute_token_distribution(
    token_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    """Count tokens by EventType name.

    Special tokens (PAD, SOS, EOS) are counted under their own keys.
    """
    counts: dict[str, int] = Counter()
    for tid in token_ids:
        if tid == tokenizer.pad_id:
            counts["PAD"] += 1
        elif tid == tokenizer.sos_id:
            counts["SOS"] += 1
        elif tid == tokenizer.eos_id:
            counts["EOS"] += 1
        else:
            event = tokenizer.decode(tid)
            counts[event.type.name] += 1
    return dict(counts)


def compute_structural_validity(
    token_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, float]:
    """Check structural properties of a generated sequence.

    Returns:
        monotonic_time: 1.0 if time is fully monotonic, 0.0 otherwise.
        has_eos: 1.0 if sequence ends with EOS, 0.0 otherwise.
        length: number of tokens (excluding SOS).
        time_span_ms: total time span in milliseconds.
    """
    ts_start, ts_end = tokenizer.event_type_range(EventType.TIME_SHIFT)
    ts_er = tokenizer.event_range[EventType.TIME_SHIFT]

    cumulative_time = 0
    monotonic = True

    for tid in token_ids:
        if ts_start <= tid <= ts_end:
            delta = ts_er.min_value + (tid - ts_start)
            cumulative_time += delta
            if cumulative_time < 0:
                monotonic = False

    has_eos = len(token_ids) > 0 and token_ids[-1] == tokenizer.eos_id

    return {
        "monotonic_time": 1.0 if monotonic else 0.0,
        "has_eos": 1.0 if has_eos else 0.0,
        "length": float(len(token_ids)),
        "time_span_ms": float(cumulative_time * MILLISECONDS_PER_STEP),
    }


def compute_object_counts(
    token_ids: list[int],
    tokenizer: Tokenizer,
) -> dict[str, int]:
    """Count hit objects (circles, sliders, spinners) in a token sequence."""
    counts: dict[str, int] = {"circle": 0, "slider": 0, "spinner": 0}
    for tid in token_ids:
        if tid < tokenizer.OFFSET:
            continue
        event = tokenizer.decode(tid)
        if event.type == EventType.CIRCLE:
            counts["circle"] += 1
        elif event.type == EventType.SLIDER_HEAD:
            counts["slider"] += 1
        elif event.type == EventType.SPINNER:
            counts["spinner"] += 1
    return counts


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two categorical distributions.

    Both p and q should be normalized (sum to 1). Missing keys are treated as 0.
    """
    all_keys = set(p) | set(q)
    if not all_keys:
        return 0.0

    divergence = 0.0
    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        m_val = 0.5 * (p_val + q_val)
        if m_val > 0:
            if p_val > 0:
                divergence += 0.5 * p_val * math.log2(p_val / m_val)
            if q_val > 0:
                divergence += 0.5 * q_val * math.log2(q_val / m_val)
    return divergence


def _normalize_distribution(counts: dict[str, int]) -> dict[str, float]:
    """Normalize integer counts to a probability distribution."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def aggregate_generation_metrics(
    generated_samples: list[list[int]],
    ground_truth_samples: list[list[int]],
    tokenizer: Tokenizer,
) -> dict[str, float]:
    """Compute aggregated metrics over multiple generated/ground-truth pairs.

    Returns a flat dict suitable for wandb logging with gen/ prefix keys.
    """
    if not generated_samples:
        return {}

    # Per-sample metrics
    all_validity: list[dict[str, float]] = []
    all_gen_counts: list[dict[str, int]] = []
    all_gt_counts: list[dict[str, int]] = []
    all_gen_dist: list[dict[str, int]] = []
    all_gt_dist: list[dict[str, int]] = []

    for gen_tokens, gt_tokens in zip(generated_samples, ground_truth_samples):
        all_validity.append(compute_structural_validity(gen_tokens, tokenizer))
        all_gen_counts.append(compute_object_counts(gen_tokens, tokenizer))
        all_gt_counts.append(compute_object_counts(gt_tokens, tokenizer))
        all_gen_dist.append(compute_token_distribution(gen_tokens, tokenizer))
        all_gt_dist.append(compute_token_distribution(gt_tokens, tokenizer))

    n = len(generated_samples)
    metrics: dict[str, float] = {}

    # Structural validity
    metrics["gen/monotonic_time_frac"] = (
        sum(v["monotonic_time"] for v in all_validity) / n
    )
    metrics["gen/eos_frac"] = sum(v["has_eos"] for v in all_validity) / n
    metrics["gen/seq_length_mean"] = sum(v["length"] for v in all_validity) / n
    metrics["gen/time_span_ms_mean"] = sum(v["time_span_ms"] for v in all_validity) / n

    # Object counts
    total_gen_objects = sum(sum(c.values()) for c in all_gen_counts)
    total_gt_objects = sum(sum(c.values()) for c in all_gt_counts)
    metrics["gen/object_count_mean"] = total_gen_objects / n
    metrics["gen/object_count_ratio"] = total_gen_objects / max(total_gt_objects, 1)

    # Token type distribution: aggregate across all samples, then compute JS divergence
    agg_gen_dist: dict[str, int] = Counter()
    agg_gt_dist: dict[str, int] = Counter()
    for d in all_gen_dist:
        for k, v in d.items():
            agg_gen_dist[k] += v
    for d in all_gt_dist:
        for k, v in d.items():
            agg_gt_dist[k] += v

    gen_norm = _normalize_distribution(agg_gen_dist)
    gt_norm = _normalize_distribution(agg_gt_dist)
    metrics["gen/js_divergence"] = _js_divergence(gen_norm, gt_norm)

    # Per-type token fractions (from generated)
    for event_type_name, frac in gen_norm.items():
        if event_type_name in ("PAD", "SOS", "EOS"):
            continue
        metrics[f"gen/dist_{event_type_name}"] = frac

    return metrics
