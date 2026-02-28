import numpy as np

from ai_osu_maps.config import DataConfig

# Vector field indices
TIME_IDX = 0
DELTA_TIME_IDX = 1
X_IDX = 2
Y_IDX = 3


def augment_objects(
    objects: np.ndarray,
    config: DataConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply data augmentation to hit object vectors.

    Augmentations:
    - Horizontal flip with 50% probability
    - Vertical flip with 50% probability
    - Speed scaling: uniformly sampled from config.speed_scale_range

    Args:
        objects: Hit object vectors of shape (N, 32).
        config: Data configuration with augmentation parameters.
        rng: Random number generator. If None, a new default one is created.

    Returns:
        Augmented copy of the object vectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    objects = objects.copy()

    # Horizontal flip: negate x coordinate (already in [-1, 1])
    if rng.random() < 0.5:
        objects[:, X_IDX] = -objects[:, X_IDX]

    # Vertical flip: negate y coordinate
    if rng.random() < 0.5:
        objects[:, Y_IDX] = -objects[:, Y_IDX]

    # Speed scaling: scale time values
    lo, hi = config.speed_scale_range
    speed_factor = rng.uniform(lo, hi)
    objects[:, TIME_IDX] *= speed_factor
    objects[:, DELTA_TIME_IDX] *= speed_factor

    return objects
