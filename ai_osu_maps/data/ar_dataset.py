from __future__ import annotations

import logging
from pathlib import Path

import torch
from slider import Beatmap
from torch.utils.data import Dataset

from ai_osu_maps.data.osu_parser_ar import events_to_tokens, parse_beatmap
from ai_osu_maps.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

AUDIO_FEATURES_FILENAME = "audio_features.pt"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}


def _find_audio_file(song_dir: Path) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        for path in song_dir.glob(f"*{ext}"):
            return path
    return None


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


class ARBeatmapDataset(Dataset):
    """Dataset for autoregressive beatmap generation.

    Loads pre-computed audio features and tokenized beatmaps.
    Each sample returns (audio_features, token_ids, conditioning_dict).

    Expects directory structure:
        dataset_dir/
            song_001/
                audio_features.pt
                audio.mp3
                easy.osu
                hard.osu
            song_002/
                ...
    """

    def __init__(
        self,
        dataset_dir: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
        max_maps: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples: list[tuple[Path, Path, Path]] = []  # (cache, osu, audio)
        dataset_path = Path(dataset_dir)
        skipped_modes = 0
        skipped_parse = 0
        skipped_no_cache = 0
        maps_used = 0

        for song_dir in sorted(dataset_path.iterdir()):
            if not song_dir.is_dir():
                continue

            if max_maps is not None and maps_used >= max_maps:
                break

            cache_path = song_dir / AUDIO_FEATURES_FILENAME
            if not cache_path.exists():
                skipped_no_cache += 1
                continue

            audio_path = _find_audio_file(song_dir)
            added_from_dir = False

            for osu_path in sorted(song_dir.glob("*.osu")):
                if not _is_osu_standard(osu_path):
                    skipped_modes += 1
                    continue
                try:
                    beatmap = Beatmap.from_path(osu_path)
                    events, _ = parse_beatmap(beatmap)
                    if len(events) == 0:
                        skipped_parse += 1
                        continue
                except Exception:
                    skipped_parse += 1
                    continue
                self.samples.append((
                    cache_path,
                    osu_path,
                    audio_path or cache_path,  # fallback
                ))
                added_from_dir = True

            if added_from_dir:
                maps_used += 1

        if skipped_modes > 0:
            logger.info("Skipped %d non-standard mode beatmaps", skipped_modes)
        if skipped_parse > 0:
            logger.info("Skipped %d unparseable beatmaps", skipped_parse)
        if skipped_no_cache > 0:
            logger.info(
                "Skipped %d song dirs without cached audio features", skipped_no_cache
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cache_path, osu_path, audio_path = self.samples[idx]

        audio_features = torch.load(cache_path, weights_only=True)  # (T, d_model)

        beatmap = Beatmap.from_path(osu_path)
        events, _ = parse_beatmap(beatmap)
        token_ids = events_to_tokens(events, self.tokenizer)

        # Truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Extract conditioning from beatmap
        difficulty = getattr(beatmap, "star_rating", None) or 5.0
        cs = beatmap.circle_size
        ar = beatmap.approach_rate
        od = beatmap.overall_difficulty
        hp = beatmap.hp_drain_rate

        return {
            "audio_features": audio_features,
            "token_ids": token_ids,
            "difficulty": torch.tensor(difficulty, dtype=torch.float32),
            "cs": torch.tensor(cs, dtype=torch.float32),
            "ar": torch.tensor(ar, dtype=torch.float32),
            "od": torch.tensor(od, dtype=torch.float32),
            "hp": torch.tensor(hp, dtype=torch.float32),
            "audio_path": str(audio_path),
        }


def ar_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
    }
