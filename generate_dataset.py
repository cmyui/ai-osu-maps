#!/usr/bin/env python3
"""Unified dataset generation pipeline.

Runs all three dataset preparation stages sequentially:
1. Download beatmapsets (.osz archives → audio + .osu files)
2. Precompute MERT audio features (audio → audio_features.pt)
3. Precompute tokenized beatmaps (.osu → beatmap_tokens.pt)

Each stage is idempotent — completed work is skipped automatically.

Usage:
    python generate_dataset.py \
        --dataset-dir dataset \
        --set-ids-file top_beatmapsets.tsv \
        --limit 10000 \
        --device cuda
"""

import argparse
import asyncio
import concurrent.futures
import logging

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
        "--force-tokens",
        action="store_true",
        help="Recompute cached beatmap tokens only",
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

    logger.info("=== Stage 1/3: Downloading beatmapsets ===")
    asyncio.run(
        download.run(
            args.dataset_dir,
            set_ids_file=args.set_ids_file,
            limit=args.limit,
            chunk_size=args.download_chunk_size,
            dry_run=args.dry_run,
        )
    )

    if args.dry_run:
        logger.info("Dry run complete, skipping precompute stages")
        return

    logger.info("=== Stage 2/3: Precomputing audio features + beatmap tokens ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        force_audio = args.force or args.force_audio
        force_tokens = args.force or args.force_tokens
        audio_future = executor.submit(
            precompute_audio.run,
            args.dataset_dir,
            device=args.device,
            force=force_audio,
            batch_size=args.audio_batch_size,
        )
        tokens_future = executor.submit(
            precompute_tokens.run, args.dataset_dir, force=force_tokens
        )
        audio_future.result()
        tokens_future.result()

    logger.info("=== Dataset generation complete ===")


if __name__ == "__main__":
    main()
