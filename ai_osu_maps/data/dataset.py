from __future__ import annotations

import bisect
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ai_osu_maps.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

AUDIO_FEATURES_FILENAME = "audio_features.pt"
BEATMAP_TOKENS_FILENAME = "beatmap_tokens.pt"

MERT_FRAME_RATE_HZ = 75


def split_song_dirs(
    dataset_dir: str,
    *,
    val_fraction: float = 0.1,
    max_maps: int | None = None,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Split song directories into train/val sets.

    Splits by song directory (not by beatmap) to prevent audio feature leakage.
    Uses a seeded shuffle for deterministic, reproducible splits.

    Returns:
        (train_dirs, val_dirs) lists of Path objects.
    """
    dataset_path = Path(dataset_dir)
    all_dirs = []

    for song_dir in sorted(dataset_path.iterdir()):
        if not song_dir.is_dir():
            continue
        if not (song_dir / AUDIO_FEATURES_FILENAME).exists():
            continue
        if not (song_dir / BEATMAP_TOKENS_FILENAME).exists():
            continue
        all_dirs.append(song_dir)

    if max_maps is not None:
        all_dirs = all_dirs[:max_maps]

    rng = random.Random(seed)
    rng.shuffle(all_dirs)

    n_val = max(1, int(len(all_dirs) * val_fraction))
    val_dirs = all_dirs[:n_val]
    train_dirs = all_dirs[n_val:]

    return train_dirs, val_dirs


class BeatmapDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for autoregressive beatmap generation.

    Loads pre-computed audio features and pre-tokenized beatmaps.

    When ``window_sec`` is set, each sample is a random time window of that
    duration (in seconds). Both the token sequence and the audio features are
    sliced to the window, giving the model aligned local context instead of
    full-song cross-attention.

    Expects directory structure:
        dataset_dir/
            song_001/
                audio_features.pt
                beatmap_tokens.pt
            song_002/
                ...

    Run precompute_audio.py and precompute_tokens.py first.
    """

    def __init__(
        self,
        dataset_dir: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
        max_maps: int | None = None,
        window_sec: float | None = None,
        song_dirs: list[Path] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.window_sec = window_sec

        self.samples: list[tuple[Path, int]] = []  # (song_dir, beatmap_index)

        if song_dirs is not None:
            # Use pre-split song directories
            for song_dir in song_dirs:
                token_cache = song_dir / BEATMAP_TOKENS_FILENAME
                beatmaps = torch.load(token_cache, weights_only=False)
                for i in range(len(beatmaps)):
                    self.samples.append((song_dir, i))
        else:
            # Scan dataset directory
            dataset_path = Path(dataset_dir)
            skipped_no_audio = 0
            skipped_no_tokens = 0
            maps_used = 0

            for song_dir in sorted(dataset_path.iterdir()):
                if not song_dir.is_dir():
                    continue

                if max_maps is not None and maps_used >= max_maps:
                    break

                audio_cache = song_dir / AUDIO_FEATURES_FILENAME
                if not audio_cache.exists():
                    skipped_no_audio += 1
                    continue

                token_cache = song_dir / BEATMAP_TOKENS_FILENAME
                if not token_cache.exists():
                    skipped_no_tokens += 1
                    continue

                beatmaps = torch.load(token_cache, weights_only=False)
                for i in range(len(beatmaps)):
                    self.samples.append((song_dir, i))

                maps_used += 1

            if skipped_no_audio > 0:
                logger.info(
                    "Skipped %d song dirs without audio features",
                    skipped_no_audio,
                )
            if skipped_no_tokens > 0:
                logger.info(
                    "Skipped %d song dirs without token cache",
                    skipped_no_tokens,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        song_dir, beatmap_idx = self.samples[idx]

        audio_features = torch.load(
            song_dir / AUDIO_FEATURES_FILENAME,
            weights_only=True,
        )  # (T_audio, d_model)

        beatmaps = torch.load(
            song_dir / BEATMAP_TOKENS_FILENAME,
            weights_only=False,
        )
        bm = beatmaps[beatmap_idx]

        token_ids = bm["token_ids"]
        token_times_ms: list[int] | None = bm.get("token_times_ms")

        if self.window_sec is not None and token_times_ms is not None:
            token_ids, audio_features = _window_sample(
                token_ids,
                token_times_ms,
                audio_features,
                self.window_sec,
                self.max_seq_len,
                self.tokenizer,
            )
        else:
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[: self.max_seq_len]
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        num_objects = bm.get("num_objects", 0)

        return {
            "audio_features": audio_features,
            "token_ids": token_ids,
            "difficulty": torch.tensor(bm["difficulty"], dtype=torch.float32),
            "cs": torch.tensor(bm["cs"], dtype=torch.float32),
            "ar": torch.tensor(bm["ar"], dtype=torch.float32),
            "od": torch.tensor(bm["od"], dtype=torch.float32),
            "hp": torch.tensor(bm["hp"], dtype=torch.float32),
            "mapper_id": torch.tensor(bm.get("mapper_id", 0), dtype=torch.long),
            "year": torch.tensor(bm.get("year", 0.0), dtype=torch.float32),
            "num_objects": torch.tensor(num_objects, dtype=torch.float32),
        }


def _window_sample(
    token_ids: list[int],
    token_times_ms: list[int],
    audio_features: torch.Tensor,
    window_sec: float,
    max_seq_len: int,
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random time window and slice tokens + audio to match.

    Returns:
        token_ids: (S,) long tensor with SOS prepended.
        audio_features: (T_window, d_model) sliced audio.
    """
    window_ms = int(window_sec * 1000)
    song_duration_ms = max(token_times_ms[-1], 1)

    # Pick a random window start (can start at 0)
    max_start = max(song_duration_ms - window_ms, 0)
    window_start_ms = random.randint(0, max_start) if max_start > 0 else 0
    window_end_ms = window_start_ms + window_ms

    # Find token range within this window using bisect
    tok_start = bisect.bisect_left(token_times_ms, window_start_ms)
    tok_end = bisect.bisect_right(token_times_ms, window_end_ms)

    # Slice tokens (exclude SOS/EOS from the original, we'll re-add SOS)
    windowed_tokens = token_ids[tok_start:tok_end]
    if len(windowed_tokens) > max_seq_len - 1:  # leave room for SOS
        windowed_tokens = windowed_tokens[: max_seq_len - 1]

    # Prepend SOS
    windowed_tokens = [tokenizer.sos_id] + windowed_tokens
    token_tensor = torch.tensor(windowed_tokens, dtype=torch.long)

    # Slice audio features to match the time window
    audio_start_frame = int(window_start_ms / 1000 * MERT_FRAME_RATE_HZ)
    audio_end_frame = int(window_end_ms / 1000 * MERT_FRAME_RATE_HZ)
    audio_start_frame = min(audio_start_frame, audio_features.shape[0])
    audio_end_frame = min(audio_end_frame, audio_features.shape[0])

    windowed_audio = audio_features[audio_start_frame:audio_end_frame]

    # Ensure at least 1 audio frame
    if windowed_audio.shape[0] == 0:
        windowed_audio = audio_features[:1]

    return token_tensor, windowed_audio


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function that pads token sequences and audio features."""
    max_tokens = max(item["token_ids"].shape[0] for item in batch)
    max_audio = max(item["audio_features"].shape[0] for item in batch)
    d_model = batch[0]["audio_features"].shape[1]

    padded_tokens = []
    token_masks = []
    padded_audio = []
    audio_masks = []

    for item in batch:
        # Pad tokens
        t = item["token_ids"]
        pad_len = max_tokens - t.shape[0]
        if pad_len > 0:
            t = torch.nn.functional.pad(t, (0, pad_len), value=0)
        padded_tokens.append(t)
        mask = torch.zeros(max_tokens, dtype=torch.bool)
        mask[: item["token_ids"].shape[0]] = True
        token_masks.append(mask)

        # Pad audio features
        f = item["audio_features"]
        if f.shape[0] < max_audio:
            pad = torch.zeros(max_audio - f.shape[0], d_model)
            f = torch.cat([f, pad], dim=0)
        padded_audio.append(f)
        amask = torch.zeros(max_audio, dtype=torch.bool)
        amask[: item["audio_features"].shape[0]] = True
        audio_masks.append(amask)

    return {
        "audio_features": torch.stack(padded_audio),
        "audio_mask": torch.stack(audio_masks),
        "token_ids": torch.stack(padded_tokens),
        "token_mask": torch.stack(token_masks),
        "difficulty": torch.stack([item["difficulty"] for item in batch]),
        "cs": torch.stack([item["cs"] for item in batch]),
        "ar": torch.stack([item["ar"] for item in batch]),
        "od": torch.stack([item["od"] for item in batch]),
        "hp": torch.stack([item["hp"] for item in batch]),
        "mapper_id": torch.stack([item["mapper_id"] for item in batch]),
        "year": torch.stack([item["year"] for item in batch]),
        "num_objects": torch.stack([item["num_objects"] for item in batch]),
    }
