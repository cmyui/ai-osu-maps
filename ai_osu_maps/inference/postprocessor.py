from __future__ import annotations

import dataclasses
import os
import uuid
import zipfile
from datetime import timedelta
from string import Template

import numpy as np
from slider import TimingPoint

from ai_osu_maps.data.event import Event
from ai_osu_maps.data.event import EventType
from ai_osu_maps.inference.slider_path import SliderPath
from ai_osu_maps.inference.timing_points_change import TimingPointsChange

OSU_FILE_EXTENSION = ".osu"
OSU_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.osu")

# Event types that represent distinct hit object / timing groups
TYPE_EVENTS = frozenset(
    {
        EventType.CIRCLE,
        EventType.SPINNER,
        EventType.SPINNER_END,
        EventType.SLIDER_HEAD,
        EventType.BEZIER_ANCHOR,
        EventType.PERFECT_ANCHOR,
        EventType.CATMULL_ANCHOR,
        EventType.RED_ANCHOR,
        EventType.LAST_ANCHOR,
        EventType.SLIDER_END,
        EventType.BEAT,
        EventType.MEASURE,
        EventType.TIMING_POINT,
        EventType.KIAI,
    },
)

BEAT_TYPES = frozenset(
    {
        EventType.BEAT,
        EventType.MEASURE,
        EventType.TIMING_POINT,
    },
)

POSITION_PRECISION = 32
X_MIN, X_MAX = 0, 512
Y_MIN, Y_MAX = 0, 384
X_GRID_COUNT = X_MAX // POSITION_PRECISION - X_MIN // POSITION_PRECISION + 1


@dataclasses.dataclass
class BeatmapConfig:
    audio_filename: str = ""
    preview_time: int = -1
    mode: int = 0

    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""
    source: str = ""
    tags: str = ""

    hp_drain_rate: float = 5
    circle_size: float = 4
    overall_difficulty: float = 8
    approach_rate: float = 9
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1

    bpm: float = 120
    offset: int = 0

    background_line: str = ""


@dataclasses.dataclass
class Group:
    event_type: EventType | None = None
    value: int = 0
    time: int = 0
    distance: int | None = None
    x: float | None = None
    y: float | None = None
    new_combo: bool = False
    hitsounds: list[int] = dataclasses.field(default_factory=list)
    samplesets: list[int] = dataclasses.field(default_factory=list)
    additions: list[int] = dataclasses.field(default_factory=list)
    volumes: list[int] = dataclasses.field(default_factory=list)


def decode_pos(pos_value: int) -> tuple[int, int]:
    gx = pos_value % X_GRID_COUNT + X_MIN // POSITION_PRECISION
    gy = pos_value // X_GRID_COUNT + Y_MIN // POSITION_PRECISION
    return gx * POSITION_PRECISION, gy * POSITION_PRECISION


def get_groups(events: list[Event]) -> list[Group]:
    """Parse event list into groups. Each group is a hit object or timing marker."""
    groups: list[Group] = []
    group = Group()

    for event in events:
        if event.type == EventType.TIME_SHIFT:
            group.time = event.value
        elif event.type == EventType.DISTANCE:
            group.distance = event.value
        elif event.type == EventType.POS:
            x, y = decode_pos(event.value)
            group.x = x
            group.y = y
        elif event.type == EventType.NEW_COMBO:
            group.new_combo = True
        elif event.type == EventType.HITSOUND:
            group.hitsounds.append((event.value % 8) * 2)
            group.samplesets.append(((event.value // 8) % 3) + 1)
            group.additions.append(((event.value // 24) % 3) + 1)
        elif event.type == EventType.VOLUME:
            group.volumes.append(event.value)
        elif event.type in TYPE_EVENTS:
            group.event_type = event.type
            group.value = event.value
            groups.append(group)
            group = Group()

    if group.event_type is not None:
        groups.append(group)

    return groups


def calculate_coordinates(
    last_pos: tuple[float, float],
    dist: float,
    num_samples: int,
    playfield_size: tuple[int, int],
) -> list[tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, num_samples)
    x_coords = last_pos[0] + dist * np.cos(angles)
    y_coords = last_pos[1] + dist * np.sin(angles)
    coordinates = [
        (x, y)
        for x, y in zip(x_coords, y_coords)
        if 0 <= x <= playfield_size[0] and 0 <= y <= playfield_size[1]
    ]
    if len(coordinates) == 0:
        return (
            [playfield_size]
            if last_pos[0] + last_pos[1] > (playfield_size[0] + playfield_size[1]) / 2
            else [(0, 0)]
        )
    return coordinates


def position_to_progress(slider_path: SliderPath, pos: np.ndarray) -> float:
    eps = 1e-4
    lr = 1.0
    t = 1.0
    for _ in range(100):
        grad = np.linalg.norm(slider_path.position_at(t) - pos) - np.linalg.norm(
            slider_path.position_at(t - eps) - pos,
        )
        t -= lr * grad
        if grad == 0 or t < 0 or t > 1:
            break
    return float(np.clip(t, 0, 1))


class Postprocessor:
    CURVE_TYPE_SHORTHAND = {
        "B": "Bezier",
        "P": "PerfectCurve",
        "C": "Catmull",
    }

    def __init__(self, bpm: float = 120.0, offset: int = 0) -> None:
        self.beat_length = 60000 / bpm
        self.offset = offset

    def generate(
        self,
        events: list[Event],
        beatmap_config: BeatmapConfig,
        timing: list[TimingPoint] | None = None,
    ) -> str:
        """Convert event list to .osu file content."""
        hit_object_strings: list[str] = []
        spinner_start: Group | None = None
        slider_head: Group | None = None
        anchor_info: list[tuple[str, float, float]] = []
        last_anchor: Group | None = None

        if timing is None:
            timing = [
                TimingPoint(
                    timedelta(milliseconds=self.offset),
                    self.beat_length,
                    4,
                    2,
                    0,
                    100,
                    None,
                    False,
                ),
            ]

        groups = get_groups(events)
        last_x, last_y = 256.0, 192.0

        for group in groups:
            hit_type = group.event_type

            # Fill in missing position from distance
            if group.distance is not None and group.x is None and group.y is None:
                coords = calculate_coordinates(
                    (last_x, last_y),
                    group.distance,
                    500,
                    (512, 384),
                )
                group.x, group.y = coords[np.random.randint(len(coords))]

            if group.x is None or group.y is None:
                group.x, group.y = last_x, last_y

            if hit_type in (
                EventType.CIRCLE,
                EventType.SLIDER_HEAD,
                EventType.BEZIER_ANCHOR,
                EventType.PERFECT_ANCHOR,
                EventType.CATMULL_ANCHOR,
                EventType.RED_ANCHOR,
                EventType.LAST_ANCHOR,
                EventType.SLIDER_END,
            ):
                last_x, last_y = group.x, group.y

            if hit_type == EventType.CIRCLE:
                hitsound = group.hitsounds[0] if group.hitsounds else 0
                sampleset = group.samplesets[0] if group.samplesets else 0
                addition = group.additions[0] if group.additions else 0
                hit_object_strings.append(
                    f"{int(round(group.x))},{int(round(group.y))},{int(round(group.time))},"
                    f"{5 if group.new_combo else 1},{hitsound},{sampleset}:{addition}:0:0:",
                )
                if group.volumes:
                    timing = self._set_volume(
                        timedelta(milliseconds=int(round(group.time))),
                        group.volumes[0],
                        timing,
                    )

            elif hit_type == EventType.SPINNER:
                spinner_start = group

            elif hit_type == EventType.SPINNER_END and spinner_start is not None:
                hitsound = group.hitsounds[0] if group.hitsounds else 0
                sampleset = group.samplesets[0] if group.samplesets else 0
                addition = group.additions[0] if group.additions else 0
                hit_object_strings.append(
                    f"256,192,{int(round(spinner_start.time))},12,{hitsound},"
                    f"{int(round(group.time))},{sampleset}:{addition}:0:0:",
                )
                if group.volumes:
                    timing = self._set_volume(
                        timedelta(milliseconds=int(round(group.time))),
                        group.volumes[0],
                        timing,
                    )
                spinner_start = None
                last_x, last_y = 256, 192

            elif hit_type == EventType.SLIDER_HEAD:
                slider_head = group

            elif hit_type == EventType.BEZIER_ANCHOR:
                anchor_info.append(("B", group.x, group.y))

            elif hit_type == EventType.PERFECT_ANCHOR:
                anchor_info.append(("P", group.x, group.y))

            elif hit_type == EventType.CATMULL_ANCHOR:
                anchor_info.append(("C", group.x, group.y))

            elif hit_type == EventType.RED_ANCHOR:
                anchor_info.append(("B", group.x, group.y))
                anchor_info.append(("B", group.x, group.y))

            elif hit_type == EventType.LAST_ANCHOR:
                anchor_info.append(("B", group.x, group.y))
                last_anchor = group

            elif (
                hit_type == EventType.SLIDER_END
                and slider_head is not None
                and last_anchor is not None
            ):
                slider_start_time = int(round(slider_head.time))
                curve_type = anchor_info[0][0] if anchor_info else "B"
                span_duration = last_anchor.time - slider_head.time
                total_duration = group.time - slider_head.time

                if total_duration <= 0 or span_duration <= 0:
                    slider_head = None
                    last_anchor = None
                    anchor_info = []
                    continue

                slides = max(int(round(total_duration / span_duration)), 1)
                span_duration = total_duration / slides

                slider_path = SliderPath(
                    self.CURVE_TYPE_SHORTHAND.get(curve_type, "Bezier"),
                    np.array(
                        [(slider_head.x, slider_head.y)]
                        + [(cp[1], cp[2]) for cp in anchor_info],
                        dtype=float,
                    ),
                )
                max_length = slider_path.get_distance()

                tp = self._timing_point_at(
                    timedelta(milliseconds=slider_start_time),
                    timing,
                )
                redline = tp if tp.parent is None else tp.parent
                last_sv = 1 if tp.parent is None else -100 / tp.ms_per_beat

                req_length = max_length * position_to_progress(
                    slider_path,
                    np.array((group.x, group.y)),
                )

                if req_length < 1e-4:
                    slider_head = None
                    last_anchor = None
                    anchor_info = []
                    continue

                sv, length = self._get_human_sv_and_length(
                    req_length,
                    max_length,
                    span_duration,
                    last_sv,
                    redline,
                    slider_head.new_combo,
                    beatmap_config.slider_multiplier,
                )

                if length > max_length * 1.5:
                    sv = (
                        max_length
                        / 100
                        / span_duration
                        * redline.ms_per_beat
                        / beatmap_config.slider_multiplier
                    )
                    sv = round(sv * 20) / 20
                    length = self._calc_length(
                        sv,
                        span_duration,
                        redline,
                        beatmap_config.slider_multiplier,
                    )

                if length > max_length + 1e-4:
                    scale = length / max_length
                    anchor_info = [
                        (
                            cp[0],
                            (cp[1] - slider_head.x) * scale + slider_head.x,
                            (cp[2] - slider_head.y) * scale + slider_head.y,
                        )
                        for cp in anchor_info
                    ]

                if sv != last_sv:
                    timing = self._set_sv(
                        timedelta(milliseconds=slider_start_time),
                        sv,
                        timing,
                    )

                node_hitsounds = (
                    slider_head.hitsounds + last_anchor.hitsounds[1:] + group.hitsounds
                )
                node_samplesets = (
                    slider_head.samplesets
                    + last_anchor.samplesets[1:]
                    + group.samplesets
                )
                node_additions = (
                    slider_head.additions + last_anchor.additions[1:] + group.additions
                )
                node_volumes = (
                    slider_head.volumes + last_anchor.volumes[1:] + group.volumes
                )

                body_hitsound = last_anchor.hitsounds[0] if last_anchor.hitsounds else 0
                body_sampleset = (
                    last_anchor.samplesets[0] if last_anchor.samplesets else 0
                )
                body_addition = last_anchor.additions[0] if last_anchor.additions else 0

                control_points = "|".join(
                    f"{int(round(cp[1]))}:{int(round(cp[2]))}" for cp in anchor_info
                )
                node_hs_str = (
                    "|".join(map(str, node_hitsounds)) if node_hitsounds else "0"
                )
                node_ss_str = (
                    "|".join(
                        f"{s}:{a}" for s, a in zip(node_samplesets, node_additions)
                    )
                    if node_samplesets
                    else "0:0"
                )

                hit_object_strings.append(
                    f"{int(round(slider_head.x))},{int(round(slider_head.y))},"
                    f"{slider_start_time},"
                    f"{6 if slider_head.new_combo else 2},"
                    f"{body_hitsound},"
                    f"{curve_type}|{control_points},"
                    f"{slides},{length},"
                    f"{node_hs_str},{node_ss_str},"
                    f"{body_sampleset}:{body_addition}:0:0:",
                )

                for i in range(min(slides + 1, len(node_volumes))):
                    t = int(round(slider_head.time + span_duration * i))
                    timing = self._set_volume(
                        timedelta(milliseconds=t),
                        node_volumes[i],
                        timing,
                    )

                slider_head = None
                last_anchor = None
                anchor_info = []

            elif hit_type == EventType.KIAI:
                timing = self._set_kiai(
                    timedelta(milliseconds=group.time),
                    bool(group.value),
                    timing,
                )

        # Remove greenlines before first red line
        if timing:
            first_red = next((tp for tp in timing if tp.parent is None), None)
            if first_red is not None:
                timing = [tp for tp in timing if tp.offset >= first_red.offset]

        # Write .osu file from template
        with open(OSU_TEMPLATE_PATH) as tf:
            template = Template(tf.read())
            hit_objects = {"hit_objects": "\n".join(hit_object_strings)}
            timing_points = {"timing_points": "\n".join(tp.pack() for tp in timing)}
            bc = dataclasses.asdict(beatmap_config)
            result = template.safe_substitute({**bc, **hit_objects, **timing_points})
            return result

    def generate_timing(self, events: list[Event]) -> list[TimingPoint]:
        """Generate timing points from BEAT/MEASURE/TIMING_POINT events."""
        markers: list[tuple[float, bool, bool]] = []  # (time, is_measure, is_redline)

        for i, event in enumerate(events):
            if event.type in BEAT_TYPES:
                # Look backwards for TIME_SHIFT
                if i > 0 and events[i - 1].type == EventType.TIME_SHIFT:
                    time = events[i - 1].value
                    markers.append(
                        (
                            time,
                            event.type == EventType.MEASURE,
                            event.type == EventType.TIMING_POINT,
                        ),
                    )

        if not markers:
            return []

        markers.sort(key=lambda x: x[0])

        timing: list[TimingPoint] = []

        # Add redlines
        for time, _, is_redline in markers:
            if is_redline:
                tp = TimingPoint(
                    timedelta(milliseconds=time),
                    1000,
                    4,
                    2,
                    0,
                    100,
                    None,
                    False,
                )
                tp_change = TimingPointsChange(tp, uninherited=True)
                timing = tp_change.add_change(timing, True)

        if not timing:
            timing = [
                TimingPoint(
                    timedelta(milliseconds=markers[0][0]),
                    1000,
                    4,
                    2,
                    0,
                    100,
                    None,
                    False,
                ),
            ]

        # Calculate BPM from markers
        last_redline_time = markers[0][0]
        beat_count = 0
        for time, is_measure, is_redline in markers[1:]:
            if is_redline:
                if beat_count > 0:
                    mpb = (time - last_redline_time) / beat_count
                    if mpb > 0:
                        redline = self._timing_point_at(
                            timedelta(milliseconds=last_redline_time),
                            timing,
                        )
                        if redline.parent is None:
                            redline.ms_per_beat = mpb
                last_redline_time = time
                beat_count = 0
            else:
                beat_count += 1

        # Handle final segment
        if beat_count > 0 and len(markers) > 1:
            final_time = markers[-1][0]
            mpb = (final_time - last_redline_time) / beat_count
            if mpb > 0:
                redline = self._timing_point_at(
                    timedelta(milliseconds=last_redline_time),
                    timing,
                )
                if redline.parent is None:
                    redline.ms_per_beat = mpb

        return timing

    @staticmethod
    def _timing_point_at(
        time: timedelta,
        timing_points: list[TimingPoint],
    ) -> TimingPoint:
        for tp in reversed(timing_points):
            if tp.offset <= time:
                return tp
        return timing_points[0]

    @staticmethod
    def _set_volume(
        time: timedelta,
        volume: int,
        timing: list[TimingPoint],
    ) -> list[TimingPoint]:
        tp = TimingPoint(time, -100, 4, 2, 0, volume, None, False)
        tp_change = TimingPointsChange(tp, volume=True)
        return tp_change.add_change(timing, True)

    @staticmethod
    def _set_sv(
        time: timedelta,
        sv: float,
        timing: list[TimingPoint],
    ) -> list[TimingPoint]:
        if sv == 0:
            return timing
        tp = TimingPoint(time, -100 / sv, 4, 2, 0, 100, None, False)
        tp_change = TimingPointsChange(tp, mpb=True)
        return tp_change.add_change(timing, True)

    @staticmethod
    def _set_kiai(
        time: timedelta,
        kiai: bool,
        timing: list[TimingPoint],
    ) -> list[TimingPoint]:
        tp = TimingPoint(time, -100, 4, 2, 0, 100, None, kiai)
        tp_change = TimingPointsChange(tp, kiai=True)
        return tp_change.add_change(timing, True)

    @staticmethod
    def _calc_length(
        sv: float,
        span_duration: float,
        redline: TimingPoint,
        slider_multiplier: float,
    ) -> float:
        return sv * span_duration * 100 / redline.ms_per_beat * slider_multiplier

    def _get_human_sv_and_length(
        self,
        req_length: float,
        max_length: float,
        span_duration: float,
        last_sv: float,
        redline: TimingPoint,
        new_combo: bool,
        slider_multiplier: float,
    ) -> tuple[float, float]:
        sv = req_length / 100 / span_duration * redline.ms_per_beat / slider_multiplier
        leniency = 0.05 if new_combo else 0.15

        if last_sv > 0 and abs(sv - last_sv) / last_sv <= leniency:
            sv = last_sv
        else:
            rounded_sv = round(sv * 20) / 20
            if rounded_sv < 0.1:
                rounded_sv = round(sv * 100) / 100
            sv = rounded_sv if rounded_sv > 1e-5 else sv

        adjusted_length = self._calc_length(
            sv,
            span_duration,
            redline,
            slider_multiplier,
        )
        return sv, adjusted_length

    @staticmethod
    def write_result(result: str, output_path: str) -> str:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        osu_path = os.path.join(
            output_path,
            f"beatmap{uuid.uuid4().hex}{OSU_FILE_EXTENSION}",
        )
        with open(osu_path, "w", encoding="utf-8-sig") as osu_file:
            osu_file.write(result)
        return osu_path

    @staticmethod
    def export_osz(osu_path: str, audio_path: str, output_path: str) -> str:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        osz_path = os.path.join(output_path, f"beatmap{uuid.uuid4().hex}.osz")
        with zipfile.ZipFile(osz_path, "w") as zipf:
            zipf.write(osu_path, os.path.basename(osu_path))
            zipf.write(audio_path, os.path.basename(audio_path))
        return osz_path
