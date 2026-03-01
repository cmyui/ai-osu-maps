#!/usr/bin/env bash
# Multi-GPU training via torchrun (DDP with NCCL backend).
# Automatically detects the number of available GPUs.
set -euo pipefail

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)

exec torchrun --nproc_per_node="$NUM_GPUS" train.py "$@"
