from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    dropout: float = 0.1
    max_seq_len: int = 2048
    mert_dim: int = 768
    text_dim: int = 384
    n_text_tokens: int = 4
    num_mappers: int = 4096


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    ema_decay: float = 0.9999
    max_epochs: int = 500
    cond_dropout: float = 0.15
    rhythm_weight: float = 3.0
    object_weight: float = 2.0
    position_weight: float = 1.5
    count_loss_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    save_every_epoch: int = 1
    num_workers: int = 4


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    cfg_scale: float = 2.0
    max_tokens: int = 8192
    lookback_tokens: int = 512
