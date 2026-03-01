"""Pre-compute tokenized beatmaps for all songs in a dataset.

Saves one beatmap_tokens.pt per song directory so training startup
can skip the expensive slider.Beatmap.from_path() parsing.

Usage:
    python -m dataset_pipeline.precompute_tokens --dataset_dir dataset
"""

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path

import torch
from slider import Beatmap

from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.osu_parser import events_to_tokens, parse_beatmap
from ai_osu_maps.data.tokenizer import MILLISECONDS_PER_STEP, Tokenizer

OBJECT_EVENT_TYPES = frozenset({EventType.CIRCLE, EventType.SLIDER_HEAD, EventType.SPINNER})


def _compute_token_times_ms(token_ids: list[int], tokenizer: Tokenizer) -> list[int]:
    """Compute cumulative time in ms for each token in the sequence.

    SOS and EOS get the current time at their position. Non-TIME_SHIFT
    tokens inherit the time of the most recent TIME_SHIFT.
    """
    ts_start = tokenizer.event_start[EventType.TIME_SHIFT]
    ts_er = tokenizer.event_range[EventType.TIME_SHIFT]
    ts_count = ts_er.max_value - ts_er.min_value + 1

    cumulative_ms = 0
    times: list[int] = []
    for tid in token_ids:
        if ts_start <= tid < ts_start + ts_count:
            delta_steps = ts_er.min_value + (tid - ts_start)
            cumulative_ms += delta_steps * MILLISECONDS_PER_STEP
        times.append(cumulative_ms)
    return times

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


def _process_song_dir(
    song_dir: Path, cache_path: Path
) -> tuple[str, int, list[str]]:
    """Process a single song directory. Returns (status, skipped_modes, error_paths)."""
    tokenizer = Tokenizer()
    beatmaps: list[dict] = []
    skipped_modes = 0
    error_paths: list[str] = []

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
            token_times_ms = _compute_token_times_ms(token_ids, tokenizer)
            num_objects = sum(1 for e in events if e.type in OBJECT_EVENT_TYPES)
            mapper_id = hash(beatmap.creator) % 4096 if beatmap.creator else 0
            beatmaps.append(
                {
                    "token_ids": token_ids,
                    "token_times_ms": token_times_ms,
                    "difficulty": getattr(beatmap, "star_rating", None) or 5.0,
                    "cs": float(beatmap.circle_size),
                    "ar": float(beatmap.approach_rate),
                    "od": float(beatmap.overall_difficulty),
                    "hp": float(beatmap.hp_drain_rate),
                    "mapper_id": mapper_id,
                    "year": 0.0,  # TODO: source actual year from osu! API or metadata
                    "num_objects": num_objects,
                }
            )
        except Exception:
            logger.error("Failed to parse %s", osu_path, exc_info=True)
            error_paths.append(str(osu_path))

    if beatmaps:
        torch.save(beatmaps, cache_path)
        return "done", skipped_modes, error_paths
    return "failed", skipped_modes, error_paths


DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) - 2)


def run(dataset_dir: str, *, force: bool = False, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
    """Pre-compute tokenized beatmaps for all songs in the dataset."""
    dataset_path = Path(dataset_dir)

    song_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    logger.info("Found %d song directories", len(song_dirs))

    # Collect work items, filtering cached upfront
    work_items: list[tuple[Path, Path]] = []
    cached = 0

    for song_dir in song_dirs:
        cache_path = song_dir / CACHE_FILENAME
        if cache_path.exists() and not force:
            cached += 1
            continue
        work_items.append((song_dir, cache_path))

    logger.info("To process: %d, already cached: %d", len(work_items), cached)

    if not work_items:
        return

    done = 0
    failed = 0
    skipped_modes = 0

    logger.info("Using %d worker processes", max_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_song_dir, song_dir, cache_path): song_dir
            for song_dir, cache_path in work_items
        }

        for future in concurrent.futures.as_completed(futures):
            song_dir = futures[future]
            try:
                status, modes_skipped, error_paths = future.result()
            except Exception:
                logger.exception("Worker crashed on %s", song_dir)
                failed += 1
                continue

            skipped_modes += modes_skipped
            for path in error_paths:
                logger.error("Failed to parse %s", path)

            if status == "done":
                done += 1
            else:
                failed += 1

            if (done + failed) % 50 == 0:
                logger.info(
                    "Progress: %d/%d done, %d failed, %d skipped modes",
                    done,
                    len(work_items),
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
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max worker processes")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = parse_args()
    run(args.dataset_dir, force=args.force, max_workers=args.max_workers)
