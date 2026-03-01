import argparse
import copy
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ai_osu_maps.config import ModelConfig, TrainingConfig
from ai_osu_maps.data.dataset import BeatmapDataset, collate_fn
from ai_osu_maps.data.tokenizer import Tokenizer
from ai_osu_maps.model.transformer import Transformer

logger = logging.getLogger(__name__)


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main_process() -> bool:
    return _get_rank() == 0


@contextmanager
def _maybe_no_sync(model: nn.Module, skip_sync: bool):
    if skip_sync and isinstance(model, DDP):
        with model.no_sync():
            yield
    else:
        yield


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train autoregressive beatmap generator"
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
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
    parser.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument(
        "--max_maps", type=int, default=None,
        help="Limit number of song directories to use",
    )
    parser.add_argument(
        "--keep_checkpoints", type=int, default=0,
        help="Number of most recent checkpoints to keep (0 = keep all)",
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
    source = _unwrap_model(model)
    for ema_param, param in zip(ema_model.parameters(), source.parameters()):
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
    model: nn.Module,
    ema_model: Transformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    audio_encoder_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model_config,
        "training_config": training_config,
    }
    if audio_encoder_state is not None:
        ckpt["audio_encoder_state_dict"] = audio_encoder_state
    torch.save(ckpt, path)
    logger.info("Saved checkpoint to %s", path)


def cleanup_old_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    """Delete oldest checkpoints, keeping the most recent `keep` files."""
    if keep <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    to_remove = checkpoints[:-keep]
    for old_ckpt in to_remove:
        old_ckpt.unlink()
        logger.info("Removed old checkpoint %s", old_ckpt)


def build_token_weight_mask(
    token_ids: torch.Tensor,
    tokenizer: Tokenizer,
    rhythm_weight: float,
    object_weight: float,
    position_weight: float,
) -> torch.Tensor:
    """Build per-token loss weight mask with higher weight on important tokens.

    Args:
        token_ids: (B, S) target token IDs
        tokenizer: Tokenizer instance
        rhythm_weight: Weight multiplier for rhythm/timing tokens
        object_weight: Weight multiplier for hit object tokens
        position_weight: Weight multiplier for position tokens

    Returns:
        weights: (B, S) loss weight mask
    """
    from ai_osu_maps.data.event import EventType

    weights = torch.ones_like(token_ids, dtype=torch.float32)

    def _range(event_type: EventType) -> tuple[int, int]:
        return tokenizer.event_type_range(event_type)

    # Rhythm/timing tokens
    ts_start, ts_end = _range(EventType.TIME_SHIFT)
    snap_start, snap_end = _range(EventType.SNAPPING)
    beat_start, beat_end = _range(EventType.BEAT)
    measure_start, measure_end = _range(EventType.MEASURE)
    tp_start, tp_end = _range(EventType.TIMING_POINT)

    is_rhythm = (
        ((token_ids >= ts_start) & (token_ids <= ts_end))
        | ((token_ids >= snap_start) & (token_ids <= snap_end))
        | ((token_ids >= beat_start) & (token_ids <= beat_end))
        | ((token_ids >= measure_start) & (token_ids <= measure_end))
        | ((token_ids >= tp_start) & (token_ids <= tp_end))
    )
    weights[is_rhythm] = rhythm_weight

    # Object tokens
    circle_start, circle_end = _range(EventType.CIRCLE)
    sh_start, sh_end = _range(EventType.SLIDER_HEAD)
    spinner_start, spinner_end = _range(EventType.SPINNER)
    se_start, se_end = _range(EventType.SLIDER_END)

    is_object = (
        ((token_ids >= circle_start) & (token_ids <= circle_end))
        | ((token_ids >= sh_start) & (token_ids <= sh_end))
        | ((token_ids >= spinner_start) & (token_ids <= spinner_end))
        | ((token_ids >= se_start) & (token_ids <= se_end))
    )
    weights[is_object] = object_weight

    # Position tokens
    pos_start, pos_end = _range(EventType.POS)
    dist_start, dist_end = _range(EventType.DISTANCE)

    is_position = (
        ((token_ids >= pos_start) & (token_ids <= pos_end))
        | ((token_ids >= dist_start) & (token_ids <= dist_end))
    )
    weights[is_position] = position_weight

    return weights


def train(args: argparse.Namespace) -> None:
    model_config = ModelConfig()
    training_config = TrainingConfig()
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
    if args.save_every is not None:
        training_config.save_every_epoch = args.save_every

    distributed = _is_distributed()
    rank = _get_rank()
    local_rank = _get_local_rank()
    is_main = _is_main_process()

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device or _default_device())

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Using device: %s (rank %d/%d)", device, rank, int(os.environ.get("WORLD_SIZE", "1")))

    # wandb (rank 0 only)
    wandb_run = None
    if args.wandb_project and is_main:
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={
                "model": vars(model_config),
                "training": vars(training_config),
            },
        )

    # Load audio encoder state (saved by precompute_audio.py)
    audio_encoder_state = None
    audio_encoder_path = Path(args.dataset_dir) / "audio_encoder.pt"
    if audio_encoder_path.exists():
        audio_encoder_state = torch.load(
            audio_encoder_path, map_location="cpu", weights_only=True,
        )
        logger.info("Loaded audio encoder state from %s", audio_encoder_path)
    else:
        logger.warning(
            "No audio_encoder.pt found in %s - inference will use random audio projection",
            args.dataset_dir,
        )

    # Tokenizer
    tokenizer = Tokenizer()
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    # Dataset
    dataset = BeatmapDataset(
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

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        dropout=model_config.dropout,
        max_seq_len=model_config.max_seq_len,
        mert_dim=model_config.mert_dim,
        text_dim=model_config.text_dim,
        n_text_tokens=model_config.n_text_tokens,
        num_mappers=model_config.num_mappers,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Transformer: %.1fM parameters", param_count / 1e6)

    # Resume (load into unwrapped model before DDP wrapping)
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        logger.info("Resumed from epoch %d, step %d", start_epoch, global_step)

    # Wrap in DDP after loading weights
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer (must reference DDP-wrapped model parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
        betas=training_config.betas,
    )

    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

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
        if sampler is not None:
            sampler.set_epoch(epoch)

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
            mapper_id = batch["mapper_id"].to(device)
            year = batch["year"].to(device)
            num_objects = batch["num_objects"].to(device)

            batch_size = token_ids.shape[0]

            # Teacher forcing: input = tokens[:-1], target = tokens[1:]
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            target_mask = token_mask[:, 1:]

            # Per-condition dropout for CFG
            drop_mask = {
                key: torch.rand(batch_size, device=device) < training_config.cond_dropout
                for key in ("difficulty", "cs", "ar", "od", "hp", "mapper", "year")
            }

            is_accum_step = (
                (batch_idx + 1) % accum_steps != 0
                and (batch_idx + 1) != len(dataloader)
            )

            with _maybe_no_sync(model, skip_sync=is_accum_step):
                with torch.amp.autocast(
                    device.type, dtype=autocast_dtype, enabled=use_autocast
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
                        drop_mask=drop_mask,
                        predict_count=True,
                    )  # (B, S-1, vocab), (B,)

                    # Cross-entropy loss with token weighting
                    loss_per_token = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        target_ids.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                        reduction="none",
                    ).reshape(target_ids.shape)

                    weights = build_token_weight_mask(
                        target_ids,
                        tokenizer,
                        training_config.rhythm_weight,
                        training_config.object_weight,
                        training_config.position_weight,
                    )

                    masked_loss = loss_per_token * weights * target_mask
                    main_loss = masked_loss.sum() / target_mask.sum().clamp(min=1)

                    # Auxiliary count loss
                    log_target = torch.log(num_objects.float().clamp(min=1))
                    count_loss = nn.functional.l1_loss(log_count_pred, log_target)

                    loss = (main_loss + training_config.count_loss_weight * count_loss) / accum_steps

                loss.backward()

            if not is_accum_step:
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

            if is_main and global_step % training_config.log_every == 0 and not is_accum_step:
                avg_loss = epoch_loss / epoch_steps
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d epoch=%d loss=%.4f count_loss=%.4f lr=%.2e",
                    global_step,
                    epoch,
                    avg_loss,
                    count_loss.item(),
                    current_lr,
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "loss": loss.item() * accum_steps,
                            "avg_loss": avg_loss,
                            "count_loss": count_loss.item(),
                            "lr": current_lr,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch, avg_epoch_loss)

        if is_main and (epoch + 1) % training_config.save_every_epoch == 0:
            ckpt_path = (
                Path(training_config.checkpoint_dir)
                / f"checkpoint_epoch_{epoch:04d}.pt"
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
                audio_encoder_state=audio_encoder_state,
            )
            if args.keep_checkpoints > 0:
                cleanup_old_checkpoints(
                    Path(training_config.checkpoint_dir), args.keep_checkpoints
                )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
