from __future__ import annotations

from ai_osu_maps.data.event import Event
from ai_osu_maps.data.event import EventRange
from ai_osu_maps.data.event import EventType

MILLISECONDS_PER_STEP = 10

# Position grid: 512x384 playfield at 32px precision
POSITION_PRECISION = 32
X_MIN, X_MAX = 0, 512
Y_MIN, Y_MAX = 0, 384
X_GRID_MIN = X_MIN // POSITION_PRECISION  # 0
X_GRID_MAX = X_MAX // POSITION_PRECISION  # 16
Y_GRID_MIN = Y_MIN // POSITION_PRECISION  # 0
Y_GRID_MAX = Y_MAX // POSITION_PRECISION  # 12
X_GRID_COUNT = X_GRID_MAX - X_GRID_MIN + 1  # 17
Y_GRID_COUNT = Y_GRID_MAX - Y_GRID_MIN + 1  # 13
POS_MAX = X_GRID_COUNT * Y_GRID_COUNT - 1  # 220


class Tokenizer:
    OFFSET = 3  # PAD=0, SOS=1, EOS=2

    EVENT_RANGES: list[EventRange] = [
        EventRange(EventType.TIME_SHIFT, -512, 512),
        EventRange(EventType.SNAPPING, 0, 16),
        EventRange(EventType.DISTANCE, 0, 640),
        EventRange(EventType.POS, 0, POS_MAX),
        EventRange(EventType.NEW_COMBO, 0, 0),
        EventRange(EventType.HITSOUND, 0, 72),
        EventRange(EventType.VOLUME, 0, 100),
        EventRange(EventType.CIRCLE, 0, 0),
        EventRange(EventType.SPINNER, 0, 0),
        EventRange(EventType.SPINNER_END, 0, 0),
        EventRange(EventType.SLIDER_HEAD, 0, 0),
        EventRange(EventType.BEZIER_ANCHOR, 0, 0),
        EventRange(EventType.PERFECT_ANCHOR, 0, 0),
        EventRange(EventType.CATMULL_ANCHOR, 0, 0),
        EventRange(EventType.RED_ANCHOR, 0, 0),
        EventRange(EventType.LAST_ANCHOR, 0, 0),
        EventRange(EventType.SLIDER_END, 0, 0),
        EventRange(EventType.BEAT, 0, 0),
        EventRange(EventType.MEASURE, 0, 0),
        EventRange(EventType.TIMING_POINT, 0, 0),
        EventRange(EventType.KIAI, 0, 1),
    ]

    def __init__(self) -> None:
        self.event_range: dict[EventType, EventRange] = {
            er.type: er for er in self.EVENT_RANGES
        }

        self.event_start: dict[EventType, int] = {}
        self.event_end: dict[EventType, int] = {}
        offset = self.OFFSET
        for er in self.EVENT_RANGES:
            self.event_start[er.type] = offset
            offset += er.max_value - er.min_value + 1
            self.event_end[er.type] = offset

        self.vocab_size: int = offset

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def sos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    def encode(self, event: Event) -> int:
        if event.type not in self.event_range:
            raise ValueError(f"unknown event type: {event.type}")

        er = self.event_range[event.type]
        offset = self.event_start[event.type]

        if not er.min_value <= event.value <= er.max_value:
            raise ValueError(
                f"event value {event.value} is not within range "
                f"[{er.min_value}, {er.max_value}] for event type {event.type}",
            )

        return offset + event.value - er.min_value

    def decode(self, token_id: int) -> Event:
        offset = self.OFFSET
        for er in self.EVENT_RANGES:
            if offset <= token_id <= offset + er.max_value - er.min_value:
                return Event(type=er.type, value=er.min_value + token_id - offset)
            offset += er.max_value - er.min_value + 1

        raise ValueError(f"token id {token_id} is not mapped to any event")

    def event_type_range(self, event_type: EventType) -> tuple[int, int]:
        if event_type not in self.event_range:
            raise ValueError(f"unknown event type: {event_type}")

        er = self.event_range[event_type]
        offset = self.event_start[event_type]
        return offset, offset + (er.max_value - er.min_value)

    def is_rhythm_token(self, token_id: int) -> bool:
        if token_id < self.OFFSET:
            return False
        event = self.decode(token_id)
        return event.type in (EventType.TIME_SHIFT, EventType.SNAPPING)

    def encode_position(self, x: int, y: int) -> int:
        gx = max(X_GRID_MIN, min(X_GRID_MAX, x // POSITION_PRECISION))
        gy = max(Y_GRID_MIN, min(Y_GRID_MAX, y // POSITION_PRECISION))
        pos_value = (gx - X_GRID_MIN) + (gy - Y_GRID_MIN) * X_GRID_COUNT
        return self.encode(Event(EventType.POS, pos_value))

    def decode_position(self, pos_value: int) -> tuple[int, int]:
        gx = pos_value % X_GRID_COUNT + X_GRID_MIN
        gy = pos_value // X_GRID_COUNT + Y_GRID_MIN
        x = gx * POSITION_PRECISION
        y = gy * POSITION_PRECISION
        return x, y
