"""Pre-compute tokenized beatmaps for all songs in a dataset.

Saves one beatmap_tokens.pt per song directory so training startup
can skip the expensive slider.Beatmap.from_path() parsing.

Usage:
    python -m dataset_pipeline.precompute_tokens --dataset_dir dataset
"""

import argparse
import logging
from pathlib import Path

import torch
from slider import Beatmap

from ai_osu_maps.data.osu_parser_ar import events_to_tokens, parse_beatmap
from ai_osu_maps.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
CACHE_FILENAME = "beatmap_tokens.pt"


def _is_osu_standard(osu_path: Path) -> bool:
    try:
        with open(osu_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Mode:"):
                    mode = line.split(":")[1].strip()
                    return mode == "0"
    except Exception:
        pass
    return True


def run(dataset_dir: str, *, force: bool = False) -> None:
    """Pre-compute tokenized beatmaps for all songs in the dataset."""
    dataset_path = Path(dataset_dir)
    tokenizer = Tokenizer()

    song_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    logger.info("Found %d song directories", len(song_dirs))

    done = 0
    cached = 0
    failed = 0
    skipped_modes = 0

    for song_dir in song_dirs:
        cache_path = song_dir / CACHE_FILENAME

        if cache_path.exists() and not force:
            cached += 1
            continue

        beatmaps: list[dict] = []

        for osu_path in sorted(song_dir.glob("*.osu")):
            if not _is_osu_standard(osu_path):
                skipped_modes += 1
                continue
            try:
                beatmap = Beatmap.from_path(osu_path)
                events, _ = parse_beatmap(beatmap)
                if len(events) == 0:
                    continue
                token_ids = events_to_tokens(events, tokenizer)
                beatmaps.append(
                    {
                        "token_ids": token_ids,
                        "difficulty": getattr(beatmap, "star_rating", None) or 5.0,
                        "cs": float(beatmap.circle_size),
                        "ar": float(beatmap.approach_rate),
                        "od": float(beatmap.overall_difficulty),
                        "hp": float(beatmap.hp_drain_rate),
                    }
                )
            except Exception:
                logger.exception("Failed to parse %s", osu_path)

        if beatmaps:
            torch.save(beatmaps, cache_path)
            done += 1
        else:
            failed += 1

        if (done + failed) % 50 == 0 and (done + failed) > 0:
            logger.info(
                "Progress: %d done, %d cached, %d failed, %d skipped modes",
                done,
                cached,
                failed,
                skipped_modes,
            )

    logger.info(
        "Complete: %d computed, %d already cached, %d failed (no valid beatmaps), %d skipped modes",
        done,
        cached,
        failed,
        skipped_modes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute tokenized beatmaps")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = parse_args()
    run(args.dataset_dir, force=args.force)
