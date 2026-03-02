#!/usr/bin/env python3
"""Unified dataset generation pipeline.

Runs all three dataset preparation stages sequentially:
1. Download beatmapsets (.osz archives → audio + .osu files)
2. Precompute MERT audio features (audio → audio_features.pt)
3. Precompute tokenized beatmaps (.osu → beatmap_tokens.pt)

Each stage is idempotent — completed work is skipped automatically.

Usage:
    # Single GPU
    python generate_dataset.py \
        --dataset-dir dataset \
        --set-ids-file top_beatmapsets.tsv \
        --limit 10000 \
        --device cuda

    # Multi-GPU (audio precompute uses all GPUs; download/tokens run on rank 0 only)
    torchrun --nproc_per_node=2 generate_dataset.py \
        --dataset-dir dataset \
        --set-ids-file top_beatmapsets.tsv \
        --limit 10000
"""

import argparse
import asyncio
import logging
import os

import torch.distributed as dist

from dataset_pipeline import download, precompute_audio, precompute_tokens

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training dataset (download + precompute audio + precompute tokens)",
    )

    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Dataset directory"
    )

    # Download stage
    parser.add_argument(
        "--set-ids-file",
        type=str,
        default=None,
        help="TSV file with beatmapset_id in first column (skips S3/Cheesegull)",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max beatmap sets to download"
    )
    parser.add_argument(
        "--download-chunk-size", type=int, default=200, help="Download chunk size"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List downloads without fetching"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if already extracted",
    )

    # Audio precompute stage
    parser.add_argument(
        "--device", type=str, default=None, help="Torch device for audio encoding"
    )
    parser.add_argument(
        "--audio-batch-size", type=int, default=8, help="Songs per audio encoding batch"
    )
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="Recompute cached audio features only",
    )

    # Token precompute stage
    parser.add_argument(
        "--token-workers",
        type=int,
        default=precompute_tokens.DEFAULT_MAX_WORKERS,
        help="Max worker processes for token precomputation",
    )
    parser.add_argument(
        "--force-tokens",
        action="store_true",
        help="Recompute cached beatmap tokens only",
    )

    # Skip stages
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download stage"
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio precompute stage"
    )
    parser.add_argument(
        "--skip-tokens", action="store_true", help="Skip token precompute stage"
    )

    # All precompute stages
    parser.add_argument(
        "--force", action="store_true", help="Recompute all cached features and tokens"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))

    if distributed:
        dist.init_process_group(backend="nccl")

    # Download and token stages are single-process (I/O and CPU bound).
    # When launched via torchrun, only rank 0 runs them; other ranks wait.
    if not args.skip_download:
        if rank == 0:
            logger.info("=== Stage 1/3: Downloading beatmapsets ===")
            force_download = args.force or args.force_download
            asyncio.run(
                download.run(
                    args.dataset_dir,
                    set_ids_file=args.set_ids_file,
                    limit=args.limit,
                    chunk_size=args.download_chunk_size,
                    dry_run=args.dry_run,
                    force=force_download,
                )
            )
        if distributed:
            dist.barrier()
        if args.dry_run:
            logger.info("Dry run complete, skipping precompute stages")
            if distributed:
                dist.destroy_process_group()
            return

    if not args.skip_audio:
        logger.info("=== Stage 2/3: Precomputing audio features ===")
        force_audio = args.force or args.force_audio
        precompute_audio.run(
            args.dataset_dir,
            device=args.device,
            force=force_audio,
            batch_size=args.audio_batch_size,
        )
        if distributed:
            dist.barrier()

    if not args.skip_tokens:
        if rank == 0:
            logger.info("=== Stage 3/3: Precomputing beatmap tokens ===")
            force_tokens = args.force or args.force_tokens
            precompute_tokens.run(
                args.dataset_dir,
                force=force_tokens,
                max_workers=args.token_workers,
            )

    logger.info("=== Dataset generation complete ===")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
