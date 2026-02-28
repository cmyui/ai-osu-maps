import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import slider
import slider.beatmap
import slider.curve

OBJECT_DIM = 32

OSU_PLAYFIELD_WIDTH = 512
OSU_PLAYFIELD_HEIGHT = 384

SNAP_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16)

# Curve type letter -> anchor type one-hot index
_CURVE_TYPE_INDEX: dict[type, int] = {
    slider.curve.MultiBezier: 0,
    slider.curve.Perfect: 1,
    slider.curve.Catmull: 2,
    slider.curve.Linear: 3,
}


def _normalize_x(x: float) -> float:
    return (x / OSU_PLAYFIELD_WIDTH) * 2.0 - 1.0


def _normalize_y(y: float) -> float:
    return (y / OSU_PLAYFIELD_HEIGHT) * 2.0 - 1.0


def _timedelta_to_ms(td: timedelta) -> float:
    return td.total_seconds() * 1000.0


def detect_snapping(time_ms: float, timing_points: list[slider.beatmap.TimingPoint]) -> int:
    """Find the closest beat snap divisor for a given time.

    Searches through timing points to find the active one, then determines
    which snap divisor best explains the given time position.

    Args:
        time_ms: Timestamp in milliseconds.
        timing_points: Sorted list of timing points from the beatmap.

    Returns:
        Index 0-7 into SNAP_DIVISORS.
    """
    # Find the active uninherited timing point
    active_tp = timing_points[0]
    for tp in timing_points:
        tp_ms = _timedelta_to_ms(tp.offset)
        if tp_ms > time_ms:
            break
        if tp.parent is None:
            active_tp = tp

    ms_per_beat = active_tp.ms_per_beat
    if ms_per_beat <= 0:
        return 0

    tp_offset_ms = _timedelta_to_ms(active_tp.offset)
    beat_position = (time_ms - tp_offset_ms) / ms_per_beat

    best_index = 0
    best_error = float("inf")
    for i, divisor in enumerate(SNAP_DIVISORS):
        snapped = round(beat_position * divisor) / divisor
        error = abs(beat_position - snapped)
        if error < best_error:
            best_error = error
            best_index = i

    return best_index


def _make_vector(
    *,
    time_norm: float,
    delta_time_norm: float,
    x: float,
    y: float,
    obj_type_index: int,
    anchor_type_index: int = -1,
    is_last_anchor: bool = False,
    is_slider_end: bool = False,
    new_combo: bool = False,
    num_repeats: int = 0,
    hitsound: int = 0,
    volume: int = 100,
    snap_index: int = 0,
) -> np.ndarray:
    """Construct a single 32-dim object vector."""
    vec = np.zeros(OBJECT_DIM, dtype=np.float32)

    vec[0] = time_norm
    vec[1] = delta_time_norm
    vec[2] = _normalize_x(x)
    vec[3] = _normalize_y(y)

    # Object type one-hot [4:8]
    if 0 <= obj_type_index < 4:
        vec[4 + obj_type_index] = 1.0

    # Anchor type one-hot [8:12]
    if 0 <= anchor_type_index < 4:
        vec[8 + anchor_type_index] = 1.0

    vec[12] = float(is_last_anchor)
    vec[13] = float(is_slider_end)
    vec[14] = float(new_combo)
    vec[15] = min(num_repeats / 10.0, 1.0)
    vec[16] = min(hitsound / 15.0, 1.0)
    vec[17] = min(volume / 100.0, 1.0)
    vec[18] = snap_index / 8.0

    return vec


def parse_osu_file(path: Path) -> tuple[np.ndarray, dict]:
    """Parse a .osu file into a continuous vector representation.

    Each hit object is converted to one or more 32-dim vectors. Circles and
    spinners produce a single vector each. Sliders are expanded into a head
    vector, one vector per control point, and an end vector.

    Args:
        path: Path to the .osu file.

    Returns:
        A tuple of (vectors, metadata) where vectors has shape (N, 32) and
        metadata is a dict with keys: difficulty, cs, ar, od, hp, mapper_id, mode.
    """
    beatmap = slider.Beatmap.from_path(path)

    metadata = {
        "difficulty": beatmap.stars(),
        "cs": beatmap.cs(),
        "ar": beatmap.ar(),
        "od": beatmap.od(),
        "hp": beatmap.hp(),
        "mapper_id": hash(beatmap.creator) % 4096,
        "mode": beatmap.mode,
    }

    hit_objects = beatmap.hit_objects()
    if not hit_objects:
        empty = np.zeros((0, OBJECT_DIM), dtype=np.float32)
        return empty, metadata

    timing_points = beatmap.timing_points

    # Compute total duration for time normalization
    last_time_ms = _timedelta_to_ms(hit_objects[-1].time)
    # Spinners may have end_time beyond the last hit object time
    for obj in hit_objects:
        if isinstance(obj, slider.beatmap.Spinner):
            end_ms = _timedelta_to_ms(obj.end_time)
            last_time_ms = max(last_time_ms, end_ms)
        elif isinstance(obj, slider.beatmap.Slider):
            end_ms = _timedelta_to_ms(obj.end_time)
            last_time_ms = max(last_time_ms, end_ms)

    total_duration = max(last_time_ms, 1.0)

    # Find max delta time for log normalization
    prev_time_ms = 0.0
    max_dt_ms = 0.0
    for obj in hit_objects:
        t_ms = _timedelta_to_ms(obj.time)
        dt = t_ms - prev_time_ms
        max_dt_ms = max(max_dt_ms, dt)
        prev_time_ms = t_ms
    log_max_dt = math.log1p(max(max_dt_ms, 1.0))

    vectors: list[np.ndarray] = []
    prev_time_ms = 0.0

    for obj in hit_objects:
        t_ms = _timedelta_to_ms(obj.time)
        dt_ms = t_ms - prev_time_ms
        time_norm = t_ms / total_duration
        delta_time_norm = math.log1p(max(dt_ms, 0.0)) / log_max_dt

        snap_index = detect_snapping(t_ms, timing_points)
        hitsound = obj.hitsound if obj.hitsound else 0
        new_combo = obj.new_combo

        if isinstance(obj, slider.beatmap.Circle):
            vec = _make_vector(
                time_norm=time_norm,
                delta_time_norm=delta_time_norm,
                x=obj.position.x,
                y=obj.position.y,
                obj_type_index=0,
                new_combo=new_combo,
                hitsound=hitsound,
                snap_index=snap_index,
            )
            vectors.append(vec)

        elif isinstance(obj, slider.beatmap.Slider):
            curve = obj.curve
            curve_type_index = _CURVE_TYPE_INDEX.get(type(curve), 0)
            control_points = curve.points

            # Slider head
            vec = _make_vector(
                time_norm=time_norm,
                delta_time_norm=delta_time_norm,
                x=obj.position.x,
                y=obj.position.y,
                obj_type_index=1,
                new_combo=new_combo,
                num_repeats=obj.repeat,
                hitsound=hitsound,
                snap_index=snap_index,
            )
            vectors.append(vec)

            # Control points (skip the first one since it matches slider head position)
            for i, point in enumerate(control_points[1:]):
                is_last = i == len(control_points) - 2
                vec = _make_vector(
                    time_norm=time_norm,
                    delta_time_norm=0.0,
                    x=point.x,
                    y=point.y,
                    obj_type_index=1,
                    anchor_type_index=curve_type_index,
                    is_last_anchor=is_last,
                    num_repeats=obj.repeat,
                    snap_index=snap_index,
                )
                vectors.append(vec)

            # Slider end (use the curve endpoint)
            end_pos = curve(1.0) if callable(curve) else control_points[-1]
            end_time_ms = _timedelta_to_ms(obj.end_time)
            end_time_norm = end_time_ms / total_duration
            vec = _make_vector(
                time_norm=end_time_norm,
                delta_time_norm=0.0,
                x=end_pos.x,
                y=end_pos.y,
                obj_type_index=1,
                is_slider_end=True,
                num_repeats=obj.repeat,
                snap_index=snap_index,
            )
            vectors.append(vec)

        elif isinstance(obj, slider.beatmap.Spinner):
            # Spinner start - spinners don't have meaningful position
            vec = _make_vector(
                time_norm=time_norm,
                delta_time_norm=delta_time_norm,
                x=OSU_PLAYFIELD_WIDTH / 2,
                y=OSU_PLAYFIELD_HEIGHT / 2,
                obj_type_index=2,
                new_combo=new_combo,
                hitsound=hitsound,
                snap_index=snap_index,
            )
            vectors.append(vec)

            # Spinner end
            end_time_ms = _timedelta_to_ms(obj.end_time)
            end_time_norm = end_time_ms / total_duration
            vec = _make_vector(
                time_norm=end_time_norm,
                delta_time_norm=0.0,
                x=OSU_PLAYFIELD_WIDTH / 2,
                y=OSU_PLAYFIELD_HEIGHT / 2,
                obj_type_index=3,
                snap_index=snap_index,
            )
            vectors.append(vec)

        prev_time_ms = t_ms

    if not vectors:
        return np.zeros((0, OBJECT_DIM), dtype=np.float32), metadata

    return np.stack(vectors, axis=0), metadata
