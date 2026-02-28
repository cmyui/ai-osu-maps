from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    # Timing
    TIME_SHIFT = "t"
    SNAPPING = "snap"
    BEAT = "beat"
    MEASURE = "measure"
    TIMING_POINT = "timing_point"

    # Positioning
    DISTANCE = "dist"
    POS = "pos"

    # Hit objects
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    SLIDER_END = "slider_end"

    # Slider anchors
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"

    # Audio / visual
    NEW_COMBO = "new_combo"
    HITSOUND = "hitsound"
    VOLUME = "volume"
    KIAI = "kiai"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"
