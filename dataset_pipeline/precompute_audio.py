"""Pre-compute MERT audio features for all songs in a dataset.

Saves one .pt file per song directory containing the audio encoder output,
so training can skip the expensive MERT forward pass.

Usage:
    python -m dataset_pipeline.precompute_audio --dataset_dir dataset --device mps
"""
import argparse
import logging
from pathlib import Path

import torch

from ai_osu_maps.model.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
CACHE_FILENAME = "audio_features.pt"


def find_audio_file(song_dir: Path) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        for path in song_dir.glob(f"*{ext}"):
            return path
    return None


def run(dataset_dir: str, *, device: str | None = None, force: bool = False) -> None:
    """Pre-compute MERT audio features for all songs in the dataset."""
    torch_device = torch.device(
        device if device
        else ("cuda" if torch.cuda.is_available() else "cpu")
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

    done = 0
    skipped = 0
    cached = 0
    failed = 0

    for song_dir in song_dirs:
        cache_path = song_dir / CACHE_FILENAME

        if cache_path.exists() and not force:
            cached += 1
            continue

        audio_path = find_audio_file(song_dir)
        if audio_path is None:
            skipped += 1
            continue

        try:
            waveform = AudioEncoder.load_audio(audio_path).to(torch_device)
            with torch.no_grad():
                features = encoder(waveform)  # (1, max_frames, d_model)
            torch.save(features.squeeze(0).cpu(), cache_path)  # (max_frames, d_model)
            done += 1
        except Exception:
            logger.exception("Failed to process %s", audio_path)
            failed += 1

        if (done + failed) % 50 == 0:
            logger.info(
                "Progress: %d done, %d cached, %d failed, %d skipped",
                done, cached, failed, skipped,
            )

    logger.info(
        "Complete: %d computed, %d already cached, %d failed, %d skipped (no audio)",
        done, cached, failed, skipped,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute MERT audio features")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    run(args.dataset_dir, device=args.device, force=args.force)
