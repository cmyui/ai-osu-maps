"""Diagnostic script to check what event types the model generates."""
import sys
from collections import Counter

import torch

from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.model.ar_transformer import ARTransformer
from ai_osu_maps.config import ARModelConfig, GenerationConfig
from ai_osu_maps.inference.sampler_ar import sample_autoregressively


def print_token_ranges(tokenizer: Tokenizer) -> None:
    print("=== Token ID ranges ===")
    for er in tokenizer.EVENT_RANGES:
        start = tokenizer.event_start[er.type]
        end = tokenizer.event_end[er.type]
        print(f"  {er.type.value:20s}: tokens {start:5d} - {end - 1:5d} ({end - start:4d} tokens)")
    print(f"  Total vocab: {tokenizer.vocab_size}")
    print()


def load_checkpoint(path: str, device: torch.device) -> tuple[ARTransformer, ARModelConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config: ARModelConfig = ckpt["model_config"]
    tokenizer = Tokenizer()
    model = ARTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dropout=0.0,
        max_seq_len=model_config.max_seq_len,
        mert_dim=model_config.mert_dim,
        text_dim=model_config.text_dim,
        n_text_tokens=model_config.n_text_tokens,
    ).to(device)

    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_config


def main() -> None:
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_ar/ar_checkpoint_epoch_0004.pt"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    device = torch.device("cpu")

    tokenizer = Tokenizer()
    print_token_ranges(tokenizer)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, model_config = load_checkpoint(checkpoint_path, device)
    print(f"Model config: d_model={model_config.d_model}, max_seq_len={model_config.max_seq_len}")

    # Create dummy audio features (zeros) - shape (1, T_audio, d_model)
    audio_features = torch.zeros(1, 64, model_config.d_model, device=device)
    audio_mask = torch.ones(1, 64, dtype=torch.bool, device=device)

    # Conditioning
    difficulty = torch.tensor([5.0], dtype=torch.float32, device=device)
    cs = torch.tensor([4.0], dtype=torch.float32, device=device)
    ar = torch.tensor([9.0], dtype=torch.float32, device=device)
    od = torch.tensor([8.0], dtype=torch.float32, device=device)
    hp = torch.tensor([5.0], dtype=torch.float32, device=device)

    # First: check raw logits from a single forward pass
    print("\n=== Raw logits check (single step from SOS) ===")
    sos_tensor = torch.tensor([[tokenizer.sos_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.generate_next_token(
            sos_tensor, audio_features, difficulty, cs, ar, od, hp,
            audio_mask=audio_mask, text_emb=None,
        )  # (1, vocab_size)

    logits_1d = logits[0]
    print(f"  Logits shape: {logits_1d.shape}")
    print(f"  Logits min={logits_1d.min().item():.4f}, max={logits_1d.max().item():.4f}, mean={logits_1d.mean().item():.4f}")
    print(f"  NaN count: {logits_1d.isnan().sum().item()}, Inf count: {logits_1d.isinf().sum().item()}")

    # Top-10 predicted tokens
    top_vals, top_ids = logits_1d.topk(10)
    print("\n  Top 10 predicted next tokens after SOS:")
    for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
        if tid < tokenizer.OFFSET:
            labels = {0: "PAD", 1: "SOS", 2: "EOS"}
            print(f"    token={tid:5d}  logit={val:8.4f}  <{labels[tid]}>")
        else:
            event = tokenizer.decode(tid)
            print(f"    token={tid:5d}  logit={val:8.4f}  {event.type.value}={event.value}")

    # Now do greedy generation (argmax, no sampling issues)
    print(f"\n=== Greedy generation ({max_tokens} tokens) ===")
    generated = [tokenizer.sos_id]
    for step in range(max_tokens):
        input_ids = generated[-model.max_seq_len:]
        token_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            step_logits = model.generate_next_token(
                token_tensor, audio_features, difficulty, cs, ar, od, hp,
                audio_mask=audio_mask, text_emb=None,
            )

        # Block PAD and SOS
        step_logits[0, tokenizer.pad_id] = float("-inf")
        step_logits[0, tokenizer.sos_id] = float("-inf")

        if step_logits[0].isnan().any() or step_logits[0].isinf().all():
            print(f"  Step {step}: NaN/Inf detected in logits, stopping")
            break

        next_token = step_logits.argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == tokenizer.eos_id:
            print(f"  EOS at step {step}")
            break

    token_ids = generated[1:]  # exclude SOS
    print(f"\nGenerated {len(token_ids)} tokens")

    # Analyze event types
    type_counter: Counter[str] = Counter()
    special_counter: Counter[str] = Counter()

    for token_id in token_ids:
        if token_id == tokenizer.pad_id:
            special_counter["PAD"] += 1
        elif token_id == tokenizer.sos_id:
            special_counter["SOS"] += 1
        elif token_id == tokenizer.eos_id:
            special_counter["EOS"] += 1
        else:
            event = tokenizer.decode(token_id)
            type_counter[event.type.value] += 1

    print("\n=== Event type distribution ===")
    for event_type, count in sorted(type_counter.items(), key=lambda x: -x[1]):
        pct = count / len(token_ids) * 100
        print(f"  {event_type:20s}: {count:5d} ({pct:5.1f}%)")

    for label, count in special_counter.items():
        pct = count / len(token_ids) * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")

    # Check for hit-object type tokens specifically
    HIT_OBJECT_TYPES = {"circle", "slider_head", "spinner"}
    hit_object_count = sum(type_counter.get(t, 0) for t in HIT_OBJECT_TYPES)
    print(f"\n=== Hit object tokens: {hit_object_count} / {len(token_ids)} ===")

    # Print first 50 tokens decoded
    print("\n=== First 50 tokens ===")
    for i, token_id in enumerate(token_ids[:50]):
        if token_id in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
            labels = {tokenizer.pad_id: "PAD", tokenizer.sos_id: "SOS", tokenizer.eos_id: "EOS"}
            print(f"  [{i:3d}] token={token_id:5d}  <{labels[token_id]}>")
        else:
            event = tokenizer.decode(token_id)
            print(f"  [{i:3d}] token={token_id:5d}  {event.type.value}={event.value}")


if __name__ == "__main__":
    main()
