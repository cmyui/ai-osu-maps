import argparse
import copy
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from ai_osu_maps.config import ARModelConfig, ARTrainingConfig
from ai_osu_maps.data.ar_dataset import ARBeatmapDataset, ar_collate_fn
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.model.ar_transformer import ARTransformer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train autoregressive beatmap generator"
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_ar")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument(
        "--max_maps", type=int, default=None,
        help="Limit number of song directories to use",
    )
    return parser.parse_args()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(param, 1.0 - decay)


def cosine_warmup_schedule(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def save_checkpoint(
    path: Path,
    model: ARTransformer,
    ema_model: ARTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_config: ARModelConfig,
    training_config: ARTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "model_config": model_config,
            "training_config": training_config,
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


def build_rhythm_weight_mask(
    token_ids: torch.Tensor, tokenizer: Tokenizer, rhythm_weight: float
) -> torch.Tensor:
    """Build per-token loss weight mask with higher weight on rhythm tokens.

    Args:
        token_ids: (B, S) target token IDs
        tokenizer: Tokenizer instance
        rhythm_weight: Weight multiplier for rhythm tokens

    Returns:
        weights: (B, S) loss weight mask
    """
    weights = torch.ones_like(token_ids, dtype=torch.float32)

    # Get rhythm token ID ranges
    ts_start, ts_end = tokenizer.event_type_range(
        tokenizer.event_range[
            next(
                er.type
                for er in tokenizer.EVENT_RANGES
                if er.type.value == "t"
            )
        ].type
    )
    snap_start, snap_end = tokenizer.event_type_range(
        tokenizer.event_range[
            next(
                er.type
                for er in tokenizer.EVENT_RANGES
                if er.type.value == "snap"
            )
        ].type
    )

    is_rhythm = (
        ((token_ids >= ts_start) & (token_ids <= ts_end))
        | ((token_ids >= snap_start) & (token_ids <= snap_end))
    )
    weights[is_rhythm] = rhythm_weight

    return weights


def train(args: argparse.Namespace) -> None:
    model_config = ARModelConfig()
    training_config = ARTrainingConfig()
    training_config.checkpoint_dir = args.checkpoint_dir

    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.lr is not None:
        training_config.lr = args.lr
    if args.max_epochs is not None:
        training_config.max_epochs = args.max_epochs
    if args.max_seq_len is not None:
        model_config.max_seq_len = args.max_seq_len
    if args.n_layers is not None:
        model_config.n_layers = args.n_layers
    if args.warmup_steps is not None:
        training_config.warmup_steps = args.warmup_steps
    if args.gradient_accumulation_steps is not None:
        training_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        training_config.log_every = args.log_every

    device = torch.device(args.device or _default_device())

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Using device: %s", device)

    # wandb
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={
                "model": vars(model_config),
                "training": vars(training_config),
            },
        )

    # Tokenizer
    tokenizer = Tokenizer()
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    # Dataset
    dataset = ARBeatmapDataset(
        args.dataset_dir,
        tokenizer,
        max_seq_len=model_config.max_seq_len,
        max_maps=args.max_maps,
    )
    logger.info("Dataset: %d beatmaps", len(dataset))
    if len(dataset) == 0:
        logger.error("No beatmaps found in %s", args.dataset_dir)
        return

    effective_batch_size = min(training_config.batch_size, len(dataset))
    num_workers = training_config.num_workers if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ar_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model
    model = ARTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dropout=model_config.dropout,
        max_seq_len=model_config.max_seq_len,
        mert_dim=model_config.mert_dim,
        text_dim=model_config.text_dim,
        n_text_tokens=model_config.n_text_tokens,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("ARTransformer: %.1fM parameters", param_count / 1e6)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        betas=training_config.betas,
    )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)

    total_steps = training_config.max_epochs * len(dataloader)
    min_lr_ratio = training_config.min_lr / training_config.lr
    accum_steps = training_config.gradient_accumulation_steps

    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(
        "Starting training: %d epochs, %d steps/epoch, batch=%d, accum=%d, lr=%.1e",
        training_config.max_epochs,
        len(dataloader),
        effective_batch_size,
        accum_steps,
        training_config.lr,
    )

    for epoch in range(start_epoch, training_config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            token_ids = batch["token_ids"].to(device)  # (B, S)
            token_mask = batch["token_mask"].to(device)  # (B, S)
            audio_features = batch["audio_features"].to(device)  # (B, T, D)
            audio_mask = batch["audio_mask"].to(device)  # (B, T)
            difficulty = batch["difficulty"].to(device)
            cs = batch["cs"].to(device)
            ar = batch["ar"].to(device)
            od = batch["od"].to(device)
            hp = batch["hp"].to(device)

            batch_size = token_ids.shape[0]

            # Teacher forcing: input = tokens[:-1], target = tokens[1:]
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            target_mask = token_mask[:, 1:]

            # Conditioning dropout for CFG
            drop_scalars = (
                torch.rand(batch_size, device=device) < training_config.cond_dropout
            )
            # Text is None during training (no text prompts in dataset)
            # drop_text would apply if we had text

            with torch.amp.autocast(
                device.type, dtype=autocast_dtype, enabled=use_autocast
            ):
                logits = model(
                    input_ids,
                    audio_features,
                    difficulty,
                    cs,
                    ar,
                    od,
                    hp,
                    audio_mask=audio_mask,
                    text_emb=None,
                    drop_scalars=drop_scalars,
                )  # (B, S-1, vocab)

                # Cross-entropy loss with rhythm weighting
                loss_per_token = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_ids.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                    reduction="none",
                ).reshape(target_ids.shape)

                # Apply rhythm weight
                weights = build_rhythm_weight_mask(
                    target_ids, tokenizer, training_config.rhythm_weight
                )

                masked_loss = loss_per_token * weights * target_mask
                loss = masked_loss.sum() / target_mask.sum().clamp(min=1)
                loss = loss / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(
                dataloader
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip_norm
                )

                # LR schedule
                lr_mult = cosine_warmup_schedule(
                    global_step,
                    training_config.warmup_steps,
                    total_steps,
                    min_lr_ratio,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = training_config.lr * lr_mult

                optimizer.step()
                optimizer.zero_grad()

                update_ema(ema_model, model, training_config.ema_decay)
                global_step += 1

            epoch_loss += loss.item() * accum_steps
            epoch_steps += 1

            if global_step % training_config.log_every == 0 and (
                (batch_idx + 1) % accum_steps == 0
            ):
                avg_loss = epoch_loss / epoch_steps
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d epoch=%d loss=%.4f lr=%.2e",
                    global_step,
                    epoch,
                    avg_loss,
                    current_lr,
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "loss": loss.item() * accum_steps,
                            "avg_loss": avg_loss,
                            "lr": current_lr,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch, avg_epoch_loss)

        if (epoch + 1) % training_config.save_every_epoch == 0:
            ckpt_path = (
                Path(training_config.checkpoint_dir)
                / f"ar_checkpoint_epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                ckpt_path,
                model,
                ema_model,
                optimizer,
                epoch,
                global_step,
                model_config,
                training_config,
            )


if __name__ == "__main__":
    train(parse_args())
