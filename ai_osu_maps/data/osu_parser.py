from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import numpy.typing as npt
from slider import Beatmap
from slider import Circle
from slider import Slider
from slider import Spinner
from slider.curve import Catmull
from slider.curve import Linear
from slider.curve import MultiBezier
from slider.curve import Perfect

from ai_osu_maps.data.event import Event
from ai_osu_maps.data.event import EventType
from ai_osu_maps.data.tokenizer import POSITION_PRECISION
from ai_osu_maps.data.tokenizer import X_MAX
from ai_osu_maps.data.tokenizer import X_MIN
from ai_osu_maps.data.tokenizer import Y_MAX
from ai_osu_maps.data.tokenizer import Y_MIN
from ai_osu_maps.data.tokenizer import Tokenizer

DIST_MIN = 0
DIST_MAX = 640


def parse_beatmap(beatmap: Beatmap) -> tuple[list[Event], list[int]]:
    """Parse an osu!standard beatmap into a sequence of Events.

    Args:
        beatmap: Beatmap object parsed from an .osu file.

    Returns:
        events: Flat list of Event objects in temporal order.
        event_times: Corresponding time in ms for each event.
    """
    hit_objects = beatmap.hit_objects(stacking=False)
    last_pos = np.array((256, 192))
    events: list[Event] = []
    event_times: list[int] = []

    for hit_object in hit_objects:
        if isinstance(hit_object, Circle):
            last_pos = _parse_circle(hit_object, events, event_times, last_pos, beatmap)
        elif isinstance(hit_object, Slider):
            last_pos = _parse_slider(hit_object, events, event_times, last_pos, beatmap)
        elif isinstance(hit_object, Spinner):
            last_pos = _parse_spinner(hit_object, events, event_times, beatmap)

    # Sort events by time (stable sort preserves within-group order)
    if len(events) > 0:
        events, event_times = zip(*sorted(zip(events, event_times), key=lambda x: x[1]))
        events, event_times = list(events), list(event_times)

    # Merge kiai events
    kiai_events = _parse_kiai(beatmap)
    events, event_times = _merge_events((events, event_times), kiai_events)

    # Merge timing events
    timing_events = _parse_timing(beatmap)
    events, event_times = _merge_events((events, event_times), timing_events)

    return events, event_times


def _add_time_event(
    time: timedelta,
    beatmap: Beatmap,
    events: list[Event],
    event_times: list[int],
    *,
    add_snap: bool = True,
) -> None:
    time_ms = int(time.total_seconds() * 1000)
    events.append(Event(EventType.TIME_SHIFT, time_ms))
    event_times.append(time_ms)

    if not add_snap:
        return

    tp = _uninherited_point_at(time, beatmap)
    if tp.ms_per_beat == 0:
        events.append(Event(EventType.SNAPPING, 0))
        event_times.append(time_ms)
        return

    beats = (time - tp.offset).total_seconds() * 1000 / tp.ms_per_beat
    snapping = 0
    if math.isfinite(beats):
        for i in range(1, 17):
            if abs(beats - round(beats * i) / i) * tp.ms_per_beat < 2:
                snapping = i
                break

    events.append(Event(EventType.SNAPPING, snapping))
    event_times.append(time_ms)


def _add_hitsound_event(
    time: timedelta,
    group_time: int,
    hitsound: int,
    addition: str,
    beatmap: Beatmap,
    events: list[Event],
    event_times: list[int],
) -> None:
    tp = _hitsound_point_at(time, beatmap)
    tp_sample_set = tp.sample_type if tp.sample_type != 0 else 2
    if not isinstance(addition, str):
        addition = "0:0"
    addition_split = addition.split(":")
    sample_set = (
        int(addition_split[0]) if addition_split[0] not in ("", "0") else tp_sample_set
    )
    addition_set = (
        int(addition_split[1])
        if len(addition_split) > 1 and addition_split[1] not in ("", "0")
        else sample_set
    )
    volume = (
        int(addition_split[3])
        if len(addition_split) > 3 and addition_split[3] not in ("", "0")
        else tp.volume
    )

    sample_set = sample_set if 0 < sample_set < 4 else 1
    addition_set = addition_set if 0 < addition_set < 4 else 1
    hitsound = hitsound & 14
    volume = int(np.clip(volume, 0, 100))

    hitsound_idx = hitsound // 2 + 8 * (sample_set - 1) + 24 * (addition_set - 1)

    events.append(Event(EventType.HITSOUND, hitsound_idx))
    events.append(Event(EventType.VOLUME, volume))
    event_times.append(group_time)
    event_times.append(group_time)


def _add_position_event(
    pos: npt.NDArray,
    last_pos: npt.NDArray,
    time: timedelta,
    events: list[Event],
    event_times: list[int],
) -> npt.NDArray:
    time_ms = int(time.total_seconds() * 1000)

    dist = int(np.clip(np.linalg.norm(pos - last_pos), DIST_MIN, DIST_MAX))
    events.append(Event(EventType.DISTANCE, dist))
    event_times.append(time_ms)

    p = pos / POSITION_PRECISION
    gx = int(np.clip(p[0], X_MIN / POSITION_PRECISION, X_MAX / POSITION_PRECISION))
    gy = int(np.clip(p[1], Y_MIN / POSITION_PRECISION, Y_MAX / POSITION_PRECISION))
    x_count = X_MAX // POSITION_PRECISION - X_MIN // POSITION_PRECISION + 1
    pos_value = int(
        (gx - X_MIN // POSITION_PRECISION)
        + (gy - Y_MIN // POSITION_PRECISION) * x_count,
    )
    events.append(Event(EventType.POS, pos_value))
    event_times.append(time_ms)

    return pos


def _add_group(
    event: EventType | Event,
    time: timedelta,
    events: list[Event],
    event_times: list[int],
    beatmap: Beatmap,
    *,
    time_event: bool = False,
    add_snap: bool = True,
    pos: npt.NDArray | None = None,
    last_pos: npt.NDArray | None = None,
    new_combo: bool = False,
    hitsound_ref_times: list[timedelta] | None = None,
    hitsounds: list[int] | None = None,
    additions: list[str] | None = None,
) -> npt.NDArray:
    time_ms = int(time.total_seconds() * 1000) if time is not None else None

    if isinstance(event, EventType):
        event = Event(event)

    if time_event:
        _add_time_event(time, beatmap, events, event_times, add_snap=add_snap)
    if pos is not None:
        last_pos = _add_position_event(pos, last_pos, time, events, event_times)
    if new_combo:
        events.append(Event(EventType.NEW_COMBO))
        event_times.append(time_ms)
    if hitsound_ref_times is not None:
        for i, ref_time in enumerate(hitsound_ref_times):
            _add_hitsound_event(
                ref_time,
                time_ms,
                hitsounds[i],
                additions[i],
                beatmap,
                events,
                event_times,
            )

    events.append(event)
    event_times.append(time_ms)

    return last_pos


def _parse_circle(
    circle: Circle,
    events: list[Event],
    event_times: list[int],
    last_pos: npt.NDArray,
    beatmap: Beatmap,
) -> npt.NDArray:
    return _add_group(
        EventType.CIRCLE,
        circle.time,
        events,
        event_times,
        beatmap,
        time_event=True,
        pos=np.array(circle.position),
        last_pos=last_pos,
        new_combo=circle.new_combo,
        hitsound_ref_times=[circle.time],
        hitsounds=[circle.hitsound],
        additions=[circle.addition],
    )


def _parse_slider(
    slider: Slider,
    events: list[Event],
    event_times: list[int],
    last_pos: npt.NDArray,
    beatmap: Beatmap,
) -> npt.NDArray:
    if len(slider.curve.points) >= 100:
        return last_pos

    last_pos = _add_group(
        EventType.SLIDER_HEAD,
        slider.time,
        events,
        event_times,
        beatmap,
        time_event=True,
        pos=np.array(slider.position),
        last_pos=last_pos,
        new_combo=slider.new_combo,
        hitsound_ref_times=[slider.time],
        hitsounds=[slider.edge_sounds[0] if len(slider.edge_sounds) > 0 else 0],
        additions=[
            slider.edge_additions[0] if len(slider.edge_additions) > 0 else "0:0",
        ],
    )

    duration: timedelta = (slider.end_time - slider.time) / slider.repeat
    control_point_count = len(slider.curve.points)

    def add_anchor(event_type: EventType, i: int, last_pos: npt.NDArray) -> npt.NDArray:
        return _add_group(
            event_type,
            slider.time + i / (control_point_count - 1) * duration,
            events,
            event_times,
            beatmap,
            pos=np.array(slider.curve.points[i]),
            last_pos=last_pos,
        )

    if isinstance(slider.curve, Linear):
        for i in range(1, control_point_count - 1):
            last_pos = add_anchor(EventType.RED_ANCHOR, i, last_pos)
    elif isinstance(slider.curve, Catmull):
        for i in range(1, control_point_count - 1):
            last_pos = add_anchor(EventType.CATMULL_ANCHOR, i, last_pos)
    elif isinstance(slider.curve, Perfect):
        for i in range(1, control_point_count - 1):
            last_pos = add_anchor(EventType.PERFECT_ANCHOR, i, last_pos)
    elif isinstance(slider.curve, MultiBezier):
        for i in range(1, control_point_count - 1):
            if slider.curve.points[i] == slider.curve.points[i + 1]:
                last_pos = add_anchor(EventType.RED_ANCHOR, i, last_pos)
            elif slider.curve.points[i] != slider.curve.points[i - 1]:
                last_pos = add_anchor(EventType.BEZIER_ANCHOR, i, last_pos)

    # Last anchor with body hitsounds and edge hitsounds
    last_pos = _add_group(
        EventType.LAST_ANCHOR,
        slider.time + duration,
        events,
        event_times,
        beatmap,
        time_event=True,
        pos=np.array(slider.curve.points[-1]),
        last_pos=last_pos,
        hitsound_ref_times=[slider.time + timedelta(milliseconds=1)]
        + [slider.time + i * duration for i in range(1, slider.repeat)],
        hitsounds=[slider.hitsound]
        + [
            slider.edge_sounds[i] if len(slider.edge_sounds) > i else 0
            for i in range(1, slider.repeat)
        ],
        additions=[slider.addition]
        + [
            slider.edge_additions[i] if len(slider.edge_additions) > i else "0:0"
            for i in range(1, slider.repeat)
        ],
    )

    try:
        end_pos = np.array(slider.curve(1))
    except (IndexError, ValueError, ZeroDivisionError):
        end_pos = np.array(slider.position)

    return _add_group(
        EventType.SLIDER_END,
        slider.end_time,
        events,
        event_times,
        beatmap,
        time_event=True,
        pos=end_pos,
        last_pos=last_pos,
        hitsound_ref_times=[slider.end_time],
        hitsounds=[slider.edge_sounds[-1] if len(slider.edge_sounds) > 0 else 0],
        additions=[
            slider.edge_additions[-1] if len(slider.edge_additions) > 0 else "0:0",
        ],
    )


def _parse_spinner(
    spinner: Spinner,
    events: list[Event],
    event_times: list[int],
    beatmap: Beatmap,
) -> npt.NDArray:
    _add_group(
        EventType.SPINNER,
        spinner.time,
        events,
        event_times,
        beatmap,
        time_event=True,
    )

    _add_group(
        EventType.SPINNER_END,
        spinner.end_time,
        events,
        event_times,
        beatmap,
        time_event=True,
        hitsound_ref_times=[spinner.end_time],
        hitsounds=[spinner.hitsound],
        additions=[spinner.addition],
    )

    return np.array((256, 192))


def _parse_kiai(beatmap: Beatmap) -> tuple[list[Event], list[int]]:
    events: list[Event] = []
    event_times: list[int] = []
    kiai = False

    for tp in beatmap.timing_points:
        if tp.kiai_mode == kiai:
            continue

        time_ms = int(tp.offset.total_seconds() * 1000)
        _add_time_event(tp.offset, beatmap, events, event_times, add_snap=True)
        events.append(Event(EventType.KIAI, int(tp.kiai_mode)))
        event_times.append(time_ms)
        kiai = tp.kiai_mode

    return events, event_times


def _parse_timing(beatmap: Beatmap) -> tuple[list[Event], list[int]]:
    timing = beatmap.timing_points
    if len(timing) == 0:
        return [], []

    events: list[Event] = []
    event_times: list[int] = []

    hit_objects = beatmap.hit_objects(stacking=False)
    if len(hit_objects) == 0:
        return [], []

    last_ho = hit_objects[-1]
    last_time = last_ho.end_time if hasattr(last_ho, "end_time") else last_ho.time

    timing_points = [tp for tp in timing if tp.bpm]

    for i, tp in enumerate(timing_points):
        next_tp = timing_points[i + 1] if i + 1 < len(timing_points) else None
        next_time = (
            next_tp.offset - timedelta(milliseconds=10) if next_tp else last_time
        )
        time = tp.offset
        measure_counter = 0
        beat_delta = timedelta(milliseconds=tp.ms_per_beat)
        while time <= next_time:
            if measure_counter == 0:
                event_type = EventType.TIMING_POINT
            elif measure_counter % tp.meter == 0:
                event_type = EventType.MEASURE
            else:
                event_type = EventType.BEAT

            _add_time_event(time, beatmap, events, event_times, add_snap=False)
            time_ms = int(time.total_seconds() * 1000)
            events.append(Event(event_type))
            event_times.append(time_ms)

            measure_counter += 1
            time += beat_delta

    return events, event_times


def _merge_events(
    events1: tuple[list[Event], list[int]],
    events2: tuple[list[Event], list[int]],
) -> tuple[list[Event], list[int]]:
    merged_events: list[Event] = []
    merged_event_times: list[int] = []
    i = 0
    j = 0

    while i < len(events1[0]) and j < len(events2[0]):
        t1 = events1[1][i]
        t2 = events2[1][j]

        if t1 <= t2:
            merged_events.append(events1[0][i])
            merged_event_times.append(t1)
            i += 1
        else:
            merged_events.append(events2[0][j])
            merged_event_times.append(t2)
            j += 1

    merged_events.extend(events1[0][i:])
    merged_events.extend(events2[0][j:])
    merged_event_times.extend(events1[1][i:])
    merged_event_times.extend(events2[1][j:])
    return merged_events, merged_event_times


def _uninherited_point_at(time: timedelta, beatmap: Beatmap):
    tp = beatmap.timing_point_at(time)
    return tp if tp.parent is None else tp.parent


def _hitsound_point_at(time: timedelta, beatmap: Beatmap):
    hs_query = time + timedelta(milliseconds=5)
    return beatmap.timing_point_at(hs_query)


MILLISECONDS_PER_STEP = 10


def _absolute_to_relative_time(events: list[Event]) -> list[Event]:
    """Convert absolute TIME_SHIFT ms values to relative deltas in 10ms steps.

    Input TIME_SHIFT values are absolute milliseconds from the parser.
    Output TIME_SHIFT values are relative deltas in 10ms steps, clamped to [-512, 512].
    """
    result: list[Event] = []
    last_time = 0

    for event in events:
        if event.type == EventType.TIME_SHIFT:
            delta_ms = event.value - last_time
            delta_steps = int(round(delta_ms / MILLISECONDS_PER_STEP))
            delta_steps = max(-512, min(512, delta_steps))
            result.append(Event(EventType.TIME_SHIFT, delta_steps))
            last_time = event.value
        else:
            result.append(event)

    return result


def _relative_to_absolute_time(events: list[Event]) -> list[Event]:
    """Convert relative TIME_SHIFT deltas (10ms steps) back to absolute milliseconds."""
    result: list[Event] = []
    current_time = 0

    for event in events:
        if event.type == EventType.TIME_SHIFT:
            current_time += event.value * MILLISECONDS_PER_STEP
            result.append(Event(EventType.TIME_SHIFT, current_time))
        else:
            result.append(event)

    return result


def events_to_tokens(events: list[Event], tokenizer: Tokenizer) -> list[int]:
    """Convert event sequence to token IDs, including SOS/EOS.

    Converts absolute TIME_SHIFT values to relative deltas before encoding.
    """
    relative_events = _absolute_to_relative_time(events)
    tokens = [tokenizer.sos_id]
    for event in relative_events:
        tokens.append(tokenizer.encode(event))
    tokens.append(tokenizer.eos_id)
    return tokens


def tokens_to_events(tokens: list[int], tokenizer: Tokenizer) -> list[Event]:
    """Convert token IDs back to events, stripping SOS/EOS/PAD.

    Converts relative TIME_SHIFT deltas back to absolute milliseconds.
    """
    events: list[Event] = []
    for token_id in tokens:
        if token_id in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id):
            continue
        events.append(tokenizer.decode(token_id))
    return _relative_to_absolute_time(events)
