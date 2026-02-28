from dataclasses import dataclass
from dataclasses import field


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 16
    obj_dim: int = 32
    window_size: int = 256
    n_registers: int = 8
    dropout: float = 0.1
    num_object_types: int = 4
    num_anchor_types: int = 4
    mert_dim: int = 768
    mert_layers: int = 13


@dataclass
class FlowConfig:
    n_steps: int = 32
    solver: str = "midpoint"
    cfg_scale: float = 2.0
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    warmup_steps: int = 5000
    min_lr: float = 1e-6
    batch_size: int = 64
    ema_decay: float = 0.9999
    max_epochs: int = 500
    cond_dropout: float = 0.2
    loss_weight_time: float = 3.0
    loss_weight_type: float = 2.0
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    save_every_epoch: int = 1


@dataclass
class DataConfig:
    dataset_dir: str = "dataset"
    sample_rate: int = 24000
    max_objects: int = 2048
    speed_scale_range: tuple[float, float] = (0.67, 1.5)
    num_workers: int = 4


@dataclass
class InferenceConfig:
    audio_path: str = ""
    output_path: str = "output.osu"
    difficulty: float = 5.0
    cs: float = 4.0
    ar: float = 4.0
    od: float = 4.0
    hp: float = 4.0
    mapper_id: int | None = None
    num_steps: int = 32
    cfg_scale: float = 2.0
    checkpoint_path: str = ""
    num_objects: int | None = None
    device: str = "cuda"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


# --- Autoregressive model configs ---


@dataclass
class ARModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    dropout: float = 0.1
    max_seq_len: int = 2048
    mert_dim: int = 768
    text_dim: int = 384
    n_text_tokens: int = 4


@dataclass
class ARTrainingConfig:
    lr: float = 3e-4
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    ema_decay: float = 0.9999
    max_epochs: int = 500
    cond_dropout: float = 0.15
    rhythm_weight: float = 3.0
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    checkpoint_dir: str = "checkpoints_ar"
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
