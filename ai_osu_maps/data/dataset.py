import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from ai_osu_maps.config import DataConfig
from ai_osu_maps.data.augmentation import augment_objects
from ai_osu_maps.data.osu_parser import parse_osu_file

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
MERT_SAMPLE_RATE = 24000


def _find_audio_file(song_dir: Path) -> Path | None:
    """Find the first audio file in a song directory."""
    for ext in AUDIO_EXTENSIONS:
        for path in song_dir.glob(f"*{ext}"):
            return path
    return None


def _is_osu_standard(osu_path: Path) -> bool:
    """Check if .osu file is osu!standard (mode 0) by reading the header."""
    try:
        with open(osu_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Mode:"):
                    mode = line.split(":")[1].strip()
                    return mode == "0"
    except Exception:
        pass
    return True  # default to standard if unreadable


class BeatmapDataset(Dataset):
    """Dataset of osu! beatmaps paired with audio.

    Expects directory structure:
        dataset_dir/
            song_001/
                audio.mp3
                easy.osu
                hard.osu
                ...
            song_002/
                ...

    Only osu!standard (mode 0) beatmaps are included.
    """

    def __init__(
        self,
        data_config: DataConfig,
        *,
        augment: bool = True,
    ) -> None:
        self.config = data_config
        self.augment = augment
        self.rng = np.random.default_rng()

        self.samples: list[tuple[Path, Path]] = []
        dataset_dir = Path(data_config.dataset_dir)
        skipped_modes = 0
        skipped_parse = 0
        skipped_audio: set[Path] = set()

        for song_dir in sorted(dataset_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            audio_path = _find_audio_file(song_dir)
            if audio_path is None:
                continue

            # Validate audio is fully loadable
            if audio_path not in skipped_audio:
                try:
                    waveform, _ = torchaudio.load(str(audio_path))
                    if waveform.numel() == 0:
                        skipped_audio.add(audio_path)
                except Exception:
                    skipped_audio.add(audio_path)

            if audio_path in skipped_audio:
                continue

            for osu_path in sorted(song_dir.glob("*.osu")):
                if not _is_osu_standard(osu_path):
                    skipped_modes += 1
                    continue
                try:
                    objects, _ = parse_osu_file(osu_path)
                    if len(objects) == 0:
                        skipped_parse += 1
                        continue
                except Exception:
                    skipped_parse += 1
                    continue
                self.samples.append((audio_path, osu_path))

        if skipped_modes > 0:
            logger.info("Skipped %d non-standard mode beatmaps", skipped_modes)
        if skipped_parse > 0:
            logger.info("Skipped %d unparseable beatmaps", skipped_parse)
        if skipped_audio:
            logger.info("Skipped %d song dirs with bad audio", len(skipped_audio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        audio_path, osu_path = self.samples[idx]

        # Load and resample audio
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != MERT_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, MERT_SAMPLE_RATE)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)  # (T_samples,)

        # Parse .osu file (pre-validated in __init__)
        objects, metadata = parse_osu_file(osu_path)

        # Apply augmentation
        if self.augment:
            objects = augment_objects(objects, self.config, self.rng)

        # Pad or truncate to max_objects
        num_objects = len(objects)
        if num_objects > self.config.max_objects:
            objects = objects[: self.config.max_objects]
            num_objects = self.config.max_objects

        padded = np.zeros((self.config.max_objects, objects.shape[1]), dtype=np.float32)
        padded[:num_objects] = objects

        # Attention mask: 1 for valid objects, 0 for padding
        mask = np.zeros(self.config.max_objects, dtype=np.float32)
        mask[:num_objects] = 1.0

        return {
            "waveform": waveform,
            "objects": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "difficulty": torch.tensor(metadata.get("difficulty", 5.0), dtype=torch.float32),
            "cs": torch.tensor(metadata.get("cs", 4.0), dtype=torch.float32),
            "ar": torch.tensor(metadata.get("ar", 4.0), dtype=torch.float32),
            "od": torch.tensor(metadata.get("od", 4.0), dtype=torch.float32),
            "hp": torch.tensor(metadata.get("hp", 4.0), dtype=torch.float32),
            "mapper_id": torch.tensor(metadata.get("mapper_id", 0), dtype=torch.long),
            "year": torch.tensor(metadata.get("year", 2020.0), dtype=torch.float32),
            "num_objects": torch.tensor(num_objects, dtype=torch.long),
        }


AUDIO_FEATURES_FILENAME = "audio_features.pt"


class CachedAudioBeatmapDataset(Dataset):
    """Dataset that loads pre-computed MERT audio features from disk.

    Expects audio_features.pt in each song directory, created by precompute_audio.py.
    Falls back to BeatmapDataset behavior if cache files are missing.
    """

    def __init__(
        self,
        data_config: DataConfig,
        *,
        augment: bool = True,
    ) -> None:
        self.config = data_config
        self.augment = augment
        self.rng = np.random.default_rng()

        self.samples: list[tuple[Path, Path]] = []
        dataset_dir = Path(data_config.dataset_dir)
        skipped_modes = 0
        skipped_parse = 0
        skipped_no_cache = 0

        for song_dir in sorted(dataset_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            cache_path = song_dir / AUDIO_FEATURES_FILENAME
            if not cache_path.exists():
                skipped_no_cache += 1
                continue

            for osu_path in sorted(song_dir.glob("*.osu")):
                if not _is_osu_standard(osu_path):
                    skipped_modes += 1
                    continue
                try:
                    objects, _ = parse_osu_file(osu_path)
                    if len(objects) == 0:
                        skipped_parse += 1
                        continue
                except Exception:
                    skipped_parse += 1
                    continue
                self.samples.append((cache_path, osu_path))

        if skipped_modes > 0:
            logger.info("Skipped %d non-standard mode beatmaps", skipped_modes)
        if skipped_parse > 0:
            logger.info("Skipped %d unparseable beatmaps", skipped_parse)
        if skipped_no_cache > 0:
            logger.info("Skipped %d song dirs without cached audio features", skipped_no_cache)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cache_path, osu_path = self.samples[idx]

        audio_features = torch.load(cache_path, weights_only=True)  # (T, d_model)

        objects, metadata = parse_osu_file(osu_path)

        if self.augment:
            objects = augment_objects(objects, self.config, self.rng)

        num_objects = len(objects)
        if num_objects > self.config.max_objects:
            objects = objects[: self.config.max_objects]
            num_objects = self.config.max_objects

        padded = np.zeros((self.config.max_objects, objects.shape[1]), dtype=np.float32)
        padded[:num_objects] = objects

        mask = np.zeros(self.config.max_objects, dtype=np.float32)
        mask[:num_objects] = 1.0

        return {
            "audio_features": audio_features,
            "objects": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "difficulty": torch.tensor(metadata.get("difficulty", 5.0), dtype=torch.float32),
            "cs": torch.tensor(metadata.get("cs", 4.0), dtype=torch.float32),
            "ar": torch.tensor(metadata.get("ar", 4.0), dtype=torch.float32),
            "od": torch.tensor(metadata.get("od", 4.0), dtype=torch.float32),
            "hp": torch.tensor(metadata.get("hp", 4.0), dtype=torch.float32),
            "mapper_id": torch.tensor(metadata.get("mapper_id", 0), dtype=torch.long),
            "year": torch.tensor(metadata.get("year", 2020.0), dtype=torch.float32),
            "num_objects": torch.tensor(num_objects, dtype=torch.long),
        }


def cached_audio_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for CachedAudioBeatmapDataset.

    Pads audio features to the max length in the batch.
    """
    max_t = max(item["audio_features"].shape[0] for item in batch)
    d_model = batch[0]["audio_features"].shape[1]

    features = []
    for item in batch:
        f = item["audio_features"]
        if f.shape[0] < max_t:
            pad = torch.zeros(max_t - f.shape[0], d_model)
            f = torch.cat([f, pad], dim=0)
        features.append(f)

    return {
        "audio_features": torch.stack(features),
        "objects": torch.stack([item["objects"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "difficulty": torch.stack([item["difficulty"] for item in batch]),
        "cs": torch.stack([item["cs"] for item in batch]),
        "ar": torch.stack([item["ar"] for item in batch]),
        "od": torch.stack([item["od"] for item in batch]),
        "hp": torch.stack([item["hp"] for item in batch]),
        "mapper_id": torch.stack([item["mapper_id"] for item in batch]),
        "year": torch.stack([item["year"] for item in batch]),
        "num_objects": torch.stack([item["num_objects"] for item in batch]),
    }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate function that pads waveforms to the max length in batch.

    All other fields are already fixed-size from __getitem__.
    """
    # Find max waveform length
    max_wav_len = max(item["waveform"].shape[0] for item in batch)

    waveforms = []
    for item in batch:
        wav = item["waveform"]
        if wav.shape[0] < max_wav_len:
            pad_size = max_wav_len - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad_size))
        waveforms.append(wav)

    return {
        "waveform": torch.stack(waveforms),
        "objects": torch.stack([item["objects"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "difficulty": torch.stack([item["difficulty"] for item in batch]),
        "cs": torch.stack([item["cs"] for item in batch]),
        "ar": torch.stack([item["ar"] for item in batch]),
        "od": torch.stack([item["od"] for item in batch]),
        "hp": torch.stack([item["hp"] for item in batch]),
        "mapper_id": torch.stack([item["mapper_id"] for item in batch]),
        "year": torch.stack([item["year"] for item in batch]),
        "num_objects": torch.stack([item["num_objects"] for item in batch]),
    }
