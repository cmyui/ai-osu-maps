# ai-osu-maps

Autoregressive transformer for generating osu! beatmaps from audio.

## Architecture

- **Transformer** (~75M params): Decoder-only transformer with cross-attention to audio features and AdaLN conditioning on scalar difficulty parameters (SR, CS, AR, OD, HP), mapper identity, and year.
  - 512 d_model, 8 heads, 12 layers, 2048 max_seq_len
  - Cross-attention every 2nd layer
  - RoPE positional encoding
  - Weight-tied token embedding ↔ lm_head
  - Per-condition dropout (independent masks per conditioning signal)
  - Mapper conditioning: `hash(creator) % 4096` embedded via `MapperEmbedder`
  - Year conditioning: `ScalarEmbedder` (TODO: year data not yet sourced, defaults to 0.0)
  - Auxiliary object count predictor head (log-space L1 loss)
- **AudioEncoder**: Frozen MERT (`m-a-p/MERT-v1-95M`) with learned layer weights and a trained projection layer (`nn.Linear(768, 512)`). Outputs fixed-length audio features at ~75Hz.
- **Tokenizer**: 2097 vocab (PAD=0, SOS=1, EOS=2, then event ranges). Events include TIME_SHIFT, SNAPPING, DISTANCE, POS, CIRCLE, SLIDER_HEAD, anchors, BEAT/MEASURE/TIMING_POINT, etc.
- **Text conditioning** (optional): sentence-transformers `all-MiniLM-L6-v2` projected to 4 cross-attention tokens.

## Audio encoder state consistency

The AudioEncoder's projection layer is randomly initialized. The same weights must be used across precomputation, training, and inference:

1. `dataset_pipeline/precompute_audio.py` saves `audio_encoder.pt` in the dataset directory on first run
2. `train.py` loads `audio_encoder.pt` from the dataset dir and embeds it in training checkpoints
3. `inference.py` loads the audio encoder state from the checkpoint (or `--audio_encoder` CLI arg)

If `audio_encoder.pt` is missing or features were precomputed with different encoder weights, inference will produce degenerate output (all TIME_SHIFT tokens). When changing encoder weights, all features must be recomputed with `--force`.

## Pipeline

### 1. Generate dataset (unified)

```bash
python generate_dataset.py \
  --dataset_dir dataset \
  --set_ids_file top_beatmapsets.tsv \
  --limit 10000 \
  --device cuda
```

Runs all three preparation stages sequentially:
1. **Download** — fetches .osz archives from mirror sites, extracts audio + .osu files
2. **Precompute audio** — runs MERT on each song's audio, saves `audio_features.pt` per directory
3. **Precompute tokens** — parses .osu files into tokenized beatmaps, saves `beatmap_tokens.pt` per directory

Each stage is idempotent (completed work is skipped), so re-running is safe and fast.

Options:
- `--set_ids_file`: TSV with beatmapset_id in first column (bypasses S3 lookup)
- `--device`: Torch device for audio encoding (default: cuda if available, else cpu)
- `--force`: Recompute cached audio features and tokens
- `--dry_run`: List downloads without fetching (skips precompute stages)
- `--limit`: Max beatmap sets to download (default: 100)
- `--chunk_size`: Download chunk size (default: 200)

Individual stages can also be run standalone:
```bash
python -m dataset_pipeline.download --dataset_dir dataset --set_ids_file top_beatmapsets.tsv --limit 10000
python -m dataset_pipeline.precompute_audio --dataset_dir dataset --device cuda
python -m dataset_pipeline.precompute_tokens --dataset_dir dataset
```

Requires `.env` with AWS credentials for S3-based discovery (not needed with `--set_ids_file`).

### 2. Train

```bash
# Single GPU
python train.py \
  --dataset_dir dataset \
  --device cuda \
  --batch_size 8 \
  --max_epochs 500 \
  --log_every 25

# Multi-GPU (DDP via torchrun)
torchrun --nproc_per_node=2 train.py \
  --dataset_dir dataset \
  --batch_size 16 \
  --max_epochs 500 \
  --log_every 25
```

- Supports multi-GPU training via `torchrun` (DDP with NCCL backend)
- When using `torchrun`, `--device` is ignored; each process uses its assigned GPU
- Loads pre-cached audio features + parses .osu files at startup
- Teacher forcing with cross-entropy loss and token weighting (rhythm 3x, objects 2x, position 1.5x; TIME_SHIFT at 1x)
- Auxiliary object count predictor head (log-space L1 loss, weight 0.1)
- Cosine warmup LR schedule, AdamW optimizer, gradient accumulation (4 steps)
- EMA weights with 0.9999 decay
- Checkpoints saved every epoch to `checkpoints/`
- `--window_sec N`: Train on random N-second time windows (slices both tokens and audio). Requires `token_times_ms` in cache (re-run `precompute_tokens` with `--force`).
- `--max_maps N`: Limit number of song directories (useful for quick experiments)
- `--resume path/to/checkpoint.pt`: Resume from a checkpoint
- `--wandb_project name`: Enable wandb logging
- Note: dataset loading is slow due to `slider.Beatmap.from_path()` parsing; use `--max_maps` for quick iterations

### 4. Inference

```bash
python inference.py \
  --audio_path song.mp3 \
  --checkpoint checkpoints/checkpoint_epoch_0149.pt \
  --difficulty 5.5 \
  --cs 4.0 --ar 9.3 --od 8.5 --hp 6.0 \
  --bpm 174.0 \
  --temperature 0.9 --timing_temperature 0.1 --top_p 0.95 \
  --max_tokens 8192 \
  --osz \
  --stream \
  --device mps
```

- `--osz`: Bundle output as .osz (includes audio file)
- `--stream`: Print each token to stderr as it's generated
- `--prompt "style text"`: Optional text conditioning
- `--cfg_scale 2.0`: Classifier-free guidance scale (only applies with text prompt)
- `--temperature 0.9`: Base sampling temperature for most tokens
- `--timing_temperature 0.1`: Near-greedy temperature for structural timing tokens (BEAT, MEASURE, TIMING_POINT). TIME_SHIFT/SNAPPING use the midpoint.
- `--no_monotonic_time`: Disable monotonic time constraint (by default, backward time jumps are masked)
- `--mapper N`: Mapper ID for style conditioning (0 = unknown)
- `--year N`: Year condition (0.0 = unknown)
- `--audio_encoder path/to/audio_encoder.pt`: Override audio encoder weights (for old checkpoints without embedded state)
- `--no_ema`: Use trained weights instead of EMA weights
- Generation takes ~40 min on MPS for 8192 tokens; much faster on CUDA

## Production deployment (Windows WSL)

Remote: `ssh windows-wifi` → `wsl -d Ubuntu-22.04`
GPU: RTX 3070 Ti (8GB), CUDA + bf16 autocast
Project: `/home/josh/ai-osu-maps/`

Commands must be sent through ssh → cmd → wsl, which causes quoting issues. Use scp to `/tmp/` on Windows, then `wsl -- bash /mnt/c/tmp/script.sh`.

Typical tmux session:
```bash
tmux new-session -d -s osumaps -n cache "... precompute_audio.py ..."
tmux new-window -t osumaps -n train "... train.py ..."
```

Keep the WSL instance alive with `exec sleep infinity` at the end of the setup script.

## Key files

```
generate_dataset.py                      # Unified dataset pipeline orchestrator
dataset_pipeline/download.py             # Beatmapset downloading
dataset_pipeline/precompute_audio.py     # Audio feature caching (MERT)
dataset_pipeline/precompute_tokens.py    # Beatmap token precomputation
train.py                                 # Training entry point
inference.py                             # Inference entry point
ai_osu_maps/config.py                   # ModelConfig, TrainingConfig, GenerationConfig
ai_osu_maps/model/transformer.py        # Transformer (decoder-only)
ai_osu_maps/model/audio_encoder.py      # MERT-based audio encoder
ai_osu_maps/model/conditioning.py       # ScalarEmbedder for AdaLN
ai_osu_maps/data/tokenizer.py           # 2097-vocab tokenizer
ai_osu_maps/data/event.py               # EventType enum, Event dataclass
ai_osu_maps/data/osu_parser.py          # Beatmap → events → tokens (and reverse)
ai_osu_maps/data/dataset.py             # PyTorch Dataset + collate function
ai_osu_maps/inference/sampler.py        # Autoregressive sampling loop
ai_osu_maps/inference/postprocessor.py  # Events → .osu file
```
