"""Pre-compute MERT audio features for all songs in a dataset.

Saves one .pt file per song directory containing the audio encoder output,
so training can skip the expensive MERT forward pass.

Usage:
    python -m dataset_pipeline.precompute_audio --dataset_dir dataset --device mps
"""

import argparse
import logging
import queue
import threading
from pathlib import Path

import torch

from ai_osu_maps.model.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
CACHE_FILENAME = "audio_features.pt"
DEFAULT_BATCH_SIZE = 8


def find_audio_file(song_dir: Path) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        for path in song_dir.glob(f"*{ext}"):
            return path
    return None


def _prefetch_audio(
    work_items: list[tuple[Path, Path, Path]],
    prefetch_queue: queue.Queue[tuple[Path, Path, torch.Tensor | None, BaseException | None] | None],
) -> None:
    """Load and resample audio files in a background thread."""
    for song_dir, cache_path, audio_path in work_items:
        try:
            waveform = AudioEncoder.load_audio(audio_path)
            prefetch_queue.put((song_dir, cache_path, waveform, None))
        except Exception as exc:
            prefetch_queue.put((song_dir, cache_path, None, exc))
    prefetch_queue.put(None)


def run(dataset_dir: str, *, device: str | None = None, force: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Pre-compute MERT audio features for all songs in the dataset."""
    torch_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", torch_device)

    dataset_path = Path(dataset_dir)
    encoder_state_path = dataset_path / "audio_encoder.pt"

    logger.info("Loading MERT audio encoder...")
    encoder = AudioEncoder(d_model=512).to(torch_device)
    encoder.requires_grad_(False)

    if encoder_state_path.exists() and not force:
        logger.info("Loading saved encoder state from %s", encoder_state_path)
        encoder.load_state_dict(
            torch.load(encoder_state_path, map_location=torch_device, weights_only=True)
        )
    else:
        logger.info("Saving encoder state to %s", encoder_state_path)
        torch.save(encoder.state_dict(), encoder_state_path)

    song_dirs = sorted(d for d in dataset_path.iterdir() if d.is_dir())
    logger.info("Found %d song directories", len(song_dirs))

    # Collect work items, filtering cached/skipped upfront
    work_items: list[tuple[Path, Path, Path]] = []
    skipped = 0
    cached = 0

    for song_dir in song_dirs:
        cache_path = song_dir / CACHE_FILENAME
        if cache_path.exists() and not force:
            cached += 1
            continue
        audio_path = find_audio_file(song_dir)
        if audio_path is None:
            skipped += 1
            continue
        work_items.append((song_dir, cache_path, audio_path))

    logger.info(
        "To process: %d, already cached: %d, skipped (no audio): %d",
        len(work_items),
        cached,
        skipped,
    )

    if not work_items:
        return

    # Prefetch audio loading in background thread so GPU doesn't idle
    prefetch_queue: queue.Queue[
        tuple[Path, Path, torch.Tensor | None, BaseException | None] | None
    ] = queue.Queue(maxsize=batch_size)
    prefetch_thread = threading.Thread(
        target=_prefetch_audio, args=(work_items, prefetch_queue), daemon=True
    )
    prefetch_thread.start()

    done = 0
    failed = 0
    finished = False

    while not finished:
        # Collect a batch of songs from the prefetch queue
        batch: list[tuple[Path, Path, torch.Tensor]] = []

        while len(batch) < batch_size:
            try:
                item = prefetch_queue.get(block=len(batch) == 0)
            except queue.Empty:
                break

            if item is None:
                finished = True
                break

            song_dir, cache_path, waveform, load_exc = item
            if load_exc is not None:
                logger.error("Failed to load %s", song_dir, exc_info=load_exc)
                failed += 1
                continue

            assert waveform is not None
            batch.append((song_dir, cache_path, waveform))

        if not batch:
            continue

        try:
            waveforms_gpu = [w.squeeze(0).to(torch_device) for _, _, w in batch]
            features_list = encoder.encode_batch(waveforms_gpu)

            for (_, cache_path, _), features in zip(batch, features_list):
                torch.save(features.cpu(), cache_path)
            done += len(batch)
        except Exception:
            logger.exception(
                "Failed to process batch of %d songs", len(batch)
            )
            failed += len(batch)

        if done % 50 < len(batch):
            logger.info(
                "Progress: %d/%d done, %d failed",
                done,
                len(work_items),
                failed,
            )

    prefetch_thread.join()

    logger.info(
        "Complete: %d computed, %d already cached, %d failed, %d skipped (no audio)",
        done,
        cached,
        failed,
        skipped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute MERT audio features")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Songs per encoding batch")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    args = parse_args()
    run(args.dataset_dir, device=args.device, force=args.force, batch_size=args.batch_size)
