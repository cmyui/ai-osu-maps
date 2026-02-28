import argparse
import copy
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from ai_osu_maps.config import Config
from ai_osu_maps.data.dataset import BeatmapDataset
from ai_osu_maps.data.dataset import collate_fn
from ai_osu_maps.model.audio_encoder import AudioEncoder
from ai_osu_maps.model.flow_transformer import FlowTransformer

logger = logging.getLogger(__name__)

COND_KEYS = ("difficulty", "cs", "ar", "od", "hp", "mapper", "year")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow-matching beatmap generator")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    config.data.dataset_dir = args.dataset_dir
    config.training.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    return config


def sample_logit_normal(
    batch_size: int,
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample timesteps from logit-normal distribution in (0, 1)."""
    z = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(z)


def build_drop_mask(
    batch_size: int,
    cond_dropout: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build per-condition dropout masks for classifier-free guidance training."""
    mask: dict[str, torch.Tensor] = {}
    for key in COND_KEYS:
        mask[key] = torch.rand(batch_size, device=device) < cond_dropout
    # Never drop timestep
    mask["timestep"] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return mask


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    """Update EMA model parameters."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(param, 1.0 - decay)


def cosine_warmup_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Cosine decay with linear warmup. Returns lr multiplier in [min_lr_ratio, 1]."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def save_checkpoint(
    path: Path,
    model: FlowTransformer,
    ema_model: FlowTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


def train(args: argparse.Namespace) -> None:
    config = build_config(args)
    device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Optional wandb
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(project=args.wandb_project, config=vars(config))

    # Dataset
    dataset = BeatmapDataset(config.data, augment=True)
    logger.info("Dataset: %d beatmaps", len(dataset))
    if len(dataset) == 0:
        logger.error("No beatmaps found in %s", config.data.dataset_dir)
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Models
    audio_encoder = AudioEncoder(d_model=config.model.d_model).to(device)
    audio_encoder.eval()

    model = FlowTransformer(config.model).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
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

    total_steps = config.training.max_epochs * len(dataloader)
    min_lr_ratio = config.training.min_lr / config.training.lr

    # Build loss weights per field
    loss_weights = torch.ones(config.model.obj_dim, device=device)
    loss_weights[0:2] = config.training.loss_weight_time  # time and delta_time
    loss_weights[4:8] = config.training.loss_weight_type  # object type one-hot

    logger.info("Starting training for %d epochs (%d steps)", config.training.max_epochs, total_steps)

    for epoch in range(start_epoch, config.training.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            # Move to device
            waveform = batch["waveform"].to(device)
            objects = batch["objects"].to(device)
            mask = batch["mask"].to(device)
            difficulty = batch["difficulty"].to(device)
            cs = batch["cs"].to(device)
            ar = batch["ar"].to(device)
            od = batch["od"].to(device)
            hp = batch["hp"].to(device)
            mapper_id = batch["mapper_id"].to(device)
            year = batch["year"].to(device)
            num_objects = batch["num_objects"].to(device)

            batch_size = objects.shape[0]

            # Extract audio features (frozen encoder)
            with torch.no_grad():
                audio_features = audio_encoder(waveform)

            # Sample flow timesteps
            t = sample_logit_normal(
                batch_size,
                config.flow.logit_normal_mean,
                config.flow.logit_normal_std,
                device,
            )

            # Build noised input: x_t = (1-t)*noise + t*data
            noise = torch.randn_like(objects)
            t_expanded = t[:, None, None]  # (B, 1, 1)
            x_t = (1.0 - t_expanded) * noise + t_expanded * objects

            # Velocity target: data - noise
            velocity_target = objects - noise

            # Conditioning dropout
            drop_mask = build_drop_mask(batch_size, config.training.cond_dropout, device)

            # Forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                velocity_pred, log_num_objects_pred = model(
                    x_t, t, audio_features,
                    difficulty, cs, ar, od, hp, mapper_id, year,
                    drop_mask,
                )

                # Weighted MSE on velocity (masked to valid objects)
                diff = (velocity_pred - velocity_target) ** 2  # (B, N, D)
                weighted_diff = diff * loss_weights.unsqueeze(0).unsqueeze(0)  # broadcast weights
                masked_diff = weighted_diff * mask.unsqueeze(-1)  # zero out padding
                velocity_loss = masked_diff.sum() / mask.sum().clamp(min=1)

                # Object count prediction loss
                log_num_objects_target = torch.log(num_objects.float().clamp(min=1))
                count_loss = torch.nn.functional.l1_loss(log_num_objects_pred, log_num_objects_target)

                total_loss = velocity_loss + 0.1 * count_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

            # LR schedule
            lr_mult = cosine_warmup_schedule(global_step, config.training.warmup_steps, total_steps, min_lr_ratio)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.training.lr * lr_mult

            optimizer.step()

            # EMA update
            update_ema(ema_model, model, config.training.ema_decay)

            global_step += 1
            epoch_loss += total_loss.item()
            epoch_steps += 1

            if global_step % config.training.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d epoch=%d loss=%.4f vel_loss=%.4f count_loss=%.4f lr=%.2e",
                    global_step, epoch, avg_loss, velocity_loss.item(), count_loss.item(), current_lr,
                )
                if wandb_run:
                    wandb_run.log({
                        "loss": total_loss.item(),
                        "velocity_loss": velocity_loss.item(),
                        "count_loss": count_loss.item(),
                        "lr": current_lr,
                        "epoch": epoch,
                    }, step=global_step)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch, avg_epoch_loss)

        if (epoch + 1) % config.training.save_every_epoch == 0:
            ckpt_path = Path(config.training.checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, model, ema_model, optimizer, epoch, global_step, config)


if __name__ == "__main__":
    train(parse_args())
