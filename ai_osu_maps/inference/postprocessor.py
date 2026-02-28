from pathlib import Path

import numpy as np

# osu! playfield dimensions
OSU_X_MAX = 512
OSU_Y_MAX = 384

# Object vector field indices (must match osu_parser.py layout)
TIME_IDX = 0
DELTA_TIME_IDX = 1
X_IDX = 2
Y_IDX = 3
TYPE_OFFSET = 4
TYPE_COUNT = 4
ANCHOR_TYPE_OFFSET = 8
ANCHOR_TYPE_COUNT = 4
IS_LAST_ANCHOR_IDX = 12
IS_SLIDER_END_IDX = 13
NEW_COMBO_IDX = 14
NUM_REPEATS_IDX = 15

# Object type indices (within [4:8] one-hot block)
# Matches osu_parser: circle=0, slider_head=1, spinner=2, spinner_end=3
TYPE_CIRCLE = 0
TYPE_SLIDER_HEAD = 1
TYPE_SPINNER = 2
TYPE_SPINNER_END = 3

# Anchor type indices (within [8:12] one-hot block)
ANCHOR_BEZIER = 0
ANCHOR_PERFECT = 1
ANCHOR_CATMULL = 2
ANCHOR_RED = 3

# osu! hit object type bitmask values
OSU_TYPE_CIRCLE = 1
OSU_TYPE_SLIDER = 2
OSU_TYPE_SPINNER = 8
OSU_NEW_COMBO = 4

CURVE_TYPE_CHARS = {
    ANCHOR_BEZIER: "B",
    ANCHOR_PERFECT: "P",
    ANCHOR_CATMULL: "C",
    ANCHOR_RED: "B",  # red anchors are bezier segment boundaries
}

DEFAULT_BPM = 120.0
DEFAULT_BEAT_LENGTH = 60000.0 / DEFAULT_BPM

OSU_TEMPLATE = """\
osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
Mode: 0
StackLeniency: 0.7

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4

[Metadata]
Title:Generated Map
TitleUnicode:Generated Map
Artist:Unknown
ArtistUnicode:Unknown
Creator:AI
Version:AI Generated ({difficulty:.1f}*)
Tags:ai generated

[Difficulty]
HPDrainRate:{hp}
CircleSize:{cs}
OverallDifficulty:{od}
ApproachRate:{ar}
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,{beat_length},4,2,0,100,1,0

[HitObjects]
{hit_objects}\
"""


def _denormalize_time(normalized_time: np.ndarray, duration_ms: float) -> np.ndarray:
    return np.clip(normalized_time, 0.0, 1.0) * duration_ms


def _denormalize_position(
    x_norm: np.ndarray, y_norm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip((x_norm + 1.0) / 2.0, 0.0, 1.0) * OSU_X_MAX
    y = np.clip((y_norm + 1.0) / 2.0, 0.0, 1.0) * OSU_Y_MAX
    return np.rint(x).astype(int), np.rint(y).astype(int)


def _format_circle(x: int, y: int, time_ms: int, *, new_combo: bool = False) -> str:
    type_bits = OSU_TYPE_CIRCLE
    if new_combo:
        type_bits |= OSU_NEW_COMBO
    return f"{x},{y},{time_ms},{type_bits},0,0:0:0:0:"


def _format_slider(
    x: int,
    y: int,
    time_ms: int,
    anchors: list[tuple[int, int]],
    curve_type: str,
    *,
    slides: int = 1,
    new_combo: bool = False,
) -> str:
    type_bits = OSU_TYPE_SLIDER
    if new_combo:
        type_bits |= OSU_NEW_COMBO
    anchor_str = "|".join(f"{ax}:{ay}" for ax, ay in anchors)
    return f"{x},{y},{time_ms},{type_bits},0,{curve_type}|{anchor_str},{slides}"


def _format_spinner(time_ms: int, end_time_ms: int, *, new_combo: bool = False) -> str:
    type_bits = OSU_TYPE_SPINNER
    if new_combo:
        type_bits |= OSU_NEW_COMBO
    x, y = OSU_X_MAX // 2, OSU_Y_MAX // 2
    return f"{x},{y},{time_ms},{type_bits},0,{end_time_ms}"


def vectors_to_osu(
    vectors: np.ndarray,
    audio_path: str,
    duration_ms: float,
    *,
    difficulty: float = 5.0,
    cs: float = 4.0,
    ar: float = 4.0,
    od: float = 4.0,
    hp: float = 4.0,
) -> str:
    """Convert generated object vectors to a complete .osu file string.

    The vector layout must match what osu_parser.py produces:
    - [0] normalized time, [1] delta time, [2-3] x/y normalized
    - [4:8] object type one-hot: circle=0, slider_head=1, spinner=2, spinner_end=3
    - [8:12] anchor type one-hot (only meaningful for slider control points)
    - [12] is_last_anchor, [13] is_slider_end, [14] new_combo
    - [15] num_repeats (normalized by /10)

    In the parser, slider heads AND their control points AND slider ends all
    share type index 1 (slider_head). They are distinguished by:
    - anchor_type fields [8:12] being set → control point
    - is_slider_end [13] > 0.5 → slider end
    - otherwise → slider head
    """
    # Sort by time
    sorted_indices = np.argsort(vectors[:, TIME_IDX])
    vectors = vectors[sorted_indices]

    # Denormalize
    times_ms = _denormalize_time(vectors[:, TIME_IDX], duration_ms)
    xs, ys = _denormalize_position(vectors[:, X_IDX], vectors[:, Y_IDX])
    times_ms_int = np.rint(times_ms).astype(int)

    # Object types
    type_logits = vectors[:, TYPE_OFFSET : TYPE_OFFSET + TYPE_COUNT]
    object_types = np.argmax(type_logits, axis=1)

    # Anchor types
    anchor_logits = vectors[:, ANCHOR_TYPE_OFFSET : ANCHOR_TYPE_OFFSET + ANCHOR_TYPE_COUNT]
    anchor_types = np.argmax(anchor_logits, axis=1)

    # Continuous fields
    is_slider_end = vectors[:, IS_SLIDER_END_IDX] > 0.5
    new_combo = vectors[:, NEW_COMBO_IDX] > 0.5
    num_repeats_raw = vectors[:, NUM_REPEATS_IDX]

    hit_object_lines: list[str] = []
    n = len(vectors)
    i = 0

    while i < n:
        obj_type = object_types[i]

        if obj_type == TYPE_CIRCLE:
            hit_object_lines.append(
                _format_circle(xs[i], ys[i], times_ms_int[i], new_combo=new_combo[i])
            )
            i += 1

        elif obj_type == TYPE_SLIDER_HEAD:
            if is_slider_end[i]:
                # Orphaned slider end, treat as circle
                hit_object_lines.append(
                    _format_circle(xs[i], ys[i], times_ms_int[i], new_combo=new_combo[i])
                )
                i += 1
                continue

            head_x, head_y, head_t = xs[i], ys[i], times_ms_int[i]
            head_new_combo = new_combo[i]
            slides = max(1, round(num_repeats_raw[i] * 10))
            curve_type_idx = ANCHOR_BEZIER

            anchors: list[tuple[int, int]] = []
            j = i + 1
            while j < n and object_types[j] == TYPE_SLIDER_HEAD:
                if is_slider_end[j]:
                    anchors.append((xs[j], ys[j]))
                    j += 1
                    break
                # Control point
                anchors.append((xs[j], ys[j]))
                curve_type_idx = anchor_types[j]
                j += 1

            if not anchors:
                hit_object_lines.append(
                    _format_circle(head_x, head_y, head_t, new_combo=head_new_combo)
                )
            else:
                curve_char = CURVE_TYPE_CHARS.get(curve_type_idx, "B")
                hit_object_lines.append(
                    _format_slider(
                        head_x, head_y, head_t, anchors, curve_char,
                        slides=slides, new_combo=head_new_combo,
                    )
                )
            i = j

        elif obj_type == TYPE_SPINNER:
            # Look for matching spinner_end
            end_time = times_ms_int[i] + 3000  # default 3s if no end found
            j = i + 1
            if j < n and object_types[j] == TYPE_SPINNER_END:
                end_time = times_ms_int[j]
                j += 1
            hit_object_lines.append(
                _format_spinner(times_ms_int[i], end_time, new_combo=new_combo[i])
            )
            i = j

        else:
            # Orphaned spinner_end or unknown type, treat as circle
            hit_object_lines.append(
                _format_circle(xs[i], ys[i], times_ms_int[i], new_combo=new_combo[i])
            )
            i += 1

    audio_filename = Path(audio_path).name

    return OSU_TEMPLATE.format(
        audio_filename=audio_filename,
        difficulty=difficulty,
        hp=hp,
        cs=cs,
        od=od,
        ar=ar,
        beat_length=DEFAULT_BEAT_LENGTH,
        hit_objects="\n".join(hit_object_lines),
    )
