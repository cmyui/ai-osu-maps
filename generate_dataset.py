"""Unified dataset generation pipeline.

Runs all three dataset preparation stages sequentially:
1. Download beatmapsets (.osz archives → audio + .osu files)
2. Precompute MERT audio features (audio → audio_features.pt)
3. Precompute tokenized beatmaps (.osu → beatmap_tokens.pt)

Each stage is idempotent — completed work is skipped automatically.

Usage:
    python generate_dataset.py \
        --dataset_dir dataset \
        --set_ids_file top_beatmapsets.tsv \
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
        "--dataset_dir", type=str, required=True, help="Dataset directory"
    )

    # Download stage
    parser.add_argument(
        "--set_ids_file",
        type=str,
        default=None,
        help="TSV file with beatmapset_id in first column (skips S3/Cheesegull)",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max beatmap sets to download"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=200, help="Download chunk size"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="List downloads without fetching"
    )

    # Audio precompute stage
    parser.add_argument(
        "--device", type=str, default=None, help="Torch device for audio encoding"
    )

    # Both precompute stages
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached features/tokens"
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
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
        )
    )

    if args.dry_run:
        logger.info("Dry run complete, skipping precompute stages")
        return

    logger.info("=== Stage 2/3: Precomputing audio features + beatmap tokens ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(
            precompute_audio.run, args.dataset_dir, device=args.device, force=args.force
        )
        tokens_future = executor.submit(
            precompute_tokens.run, args.dataset_dir, force=args.force
        )
        audio_future.result()
        tokens_future.result()

    logger.info("=== Dataset generation complete ===")


if __name__ == "__main__":
    main()
