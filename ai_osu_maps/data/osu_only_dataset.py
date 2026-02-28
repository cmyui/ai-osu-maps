"""Dataset that loads only .osu files (no audio). For training without audio."""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ai_osu_maps.config import DataConfig
from ai_osu_maps.data.augmentation import augment_objects
from ai_osu_maps.data.osu_parser import parse_osu_file

logger = logging.getLogger(__name__)


class OsuOnlyDataset(Dataset):
    """Dataset of parsed .osu files without audio.

    Scans a directory for .osu files and returns parsed object vectors
    with metadata. Audio features must be synthesized externally.
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

        dataset_dir = Path(data_config.dataset_dir)
        self.osu_files: list[Path] = []
        skipped = 0
        for osu_path in sorted(dataset_dir.glob("*.osu")):
            try:
                objects, _ = parse_osu_file(osu_path)
                if len(objects) == 0:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            self.osu_files.append(osu_path)

        if skipped > 0:
            logger.info("Skipped %d unparseable .osu files", skipped)

    def __len__(self) -> int:
        return len(self.osu_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        osu_path = self.osu_files[idx]

        # Pre-validated in __init__
        objects, metadata = parse_osu_file(osu_path)

        if self.augment:
            objects = augment_objects(objects, self.config, self.rng)

        num_objects = len(objects)
        if num_objects > self.config.max_objects:
            objects = objects[: self.config.max_objects]
            num_objects = self.config.max_objects

        padded = np.zeros((self.config.max_objects, 32), dtype=np.float32)
        padded[:num_objects] = objects

        mask = np.zeros(self.config.max_objects, dtype=np.float32)
        mask[:num_objects] = 1.0

        return {
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


def osu_only_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
