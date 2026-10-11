from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from animaleyes.camera import Frame, FrameBuffer
from animaleyes.config import ConfigStore
from animaleyes.feeder import FeederError
from animaleyes.machine import Machine
from animaleyes.motion import MotionDetector
from animaleyes.notify import LogNotifier
from animaleyes.store import Store
from animaleyes.vision import FeedingVerdict, Verdict

TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b"
    "0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432"
    "ffc0000b080001000101011100ffc40014000100000000000000000000000000000009ffc40014100100000000"
    "00000000000000000000000000ffda0008010100003f00d2cf20ffd9"
)


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


GRRR = Verdict(animal="grrr", confidence=0.95, at_bowl=True, reason="small black dog at bowl")
BOWIE = Verdict(animal="bowie", confidence=0.9, at_bowl=True, reason="big blonde dog")
CAT = Verdict(animal="cat", confidence=0.9, at_bowl=True, reason="cat")
UNSURE = Verdict(animal="unsure", confidence=0.2, at_bowl=False, reason="dark")
EATING = FeedingVerdict(grrr_at_bowl=True, bowl="food", reason="eating")
GONE_EMPTY = FeedingVerdict(grrr_at_bowl=False, bowl="empty", reason="gone, bowl empty")
GONE_FULL = FeedingVerdict(grrr_at_bowl=False, bowl="food", reason="gone, bowl full")
HAS_FOOD = FeedingVerdict(grrr_at_bowl=True, bowl="food", reason="plate has food")
NO_FOOD = FeedingVerdict(grrr_at_bowl=False, bowl="empty", reason="plate is empty")
FOOD_UNSURE = FeedingVerdict(grrr_at_bowl=False, bowl="unsure", reason="can't tell")


class FakeLLM:
    def __init__(self) -> None:
        self.identify_result: Verdict = UNSURE
        self.feeding_result: FeedingVerdict = EATING
        self.verify_result: FeedingVerdict = HAS_FOOD
        self.verify_results: list[FeedingVerdict] | None = None  # consumed in order if set
        self.identify_calls = 0
        self.feeding_calls = 0
        self.verify_calls = 0

    def identify(self, frames):
        self.identify_calls += 1
        return self.identify_result

    def feeding_check(self, frames):
        self.feeding_calls += 1
        return self.feeding_result

    def verify_food(self, frames):
        self.verify_calls += 1
        if self.verify_results:
            # Consume in order; the last entry sticks so a steady state is easy to express.
            return (
                self.verify_results.pop(0)
                if len(self.verify_results) > 1
                else self.verify_results[0]
            )  # noqa: E501
        return self.verify_result


class FakeFeeder:
    def __init__(self, plate: int = 1):
        self.plate = plate
        self.calls: list[str] = []
        self.fail_open = False
        self.open_error = "cloud said no"  # the FeederError text when fail_open is set
        self.fail_close = False
        self.nothing_to_close = False

    def current_plate(self) -> int:
        return self.plate

    def rotate(self) -> None:
        self.plate = self.plate % 3 + 1
        self.calls.append("rotate")

    def open_now(self, plate: int) -> None:
        if self.fail_open:
            raise FeederError(self.open_error)
        self.calls.append(f"open:{plate}")
        self._open = True

    def close(self) -> bool:
        if self.fail_close:
            raise FeederError("close failed")
        if self.nothing_to_close:
            return False  # the feeder's cloud says no lid is open; nothing is sent
        self.calls.append("close")
        self._open = False
        return True

    def manual_feed_active(self) -> bool:
        return getattr(self, "_open", False)

    @property
    def opens(self) -> list[str]:
        return [c for c in self.calls if c.startswith("open")]


class Harness:
    """A Machine wired to fakes, with helpers that read like the scenarios in the spec."""

    def __init__(self, tmp_path: Path, start: datetime = datetime(2026, 9, 24, 22, 0)):
        self.tmp_path = tmp_path
        self.clock = FakeClock(start)
        self.config = ConfigStore(tmp_path / "config.toml")
        self.config.update({"DRY_RUN": False, "ACTIVE_START": "21:00", "ACTIVE_END": "06:00"})
        self.store = Store(tmp_path / "state.sqlite")
        self.frames = FrameBuffer()
        self.motion = MotionDetector()
        self.llm = FakeLLM()
        self.feeder = FakeFeeder()
        self.notifier = LogNotifier()
        self.capture_gate = None  # tests set this to a FakeLLM to exercise capture mode
        self.power = None  # tests set this to a FakePower to exercise power cycling
        self.machine = self.new_machine()

    def new_machine(self) -> Machine:
        return Machine(
            config=self.config,
            store=self.store,
            frames=self.frames,
            motion=self.motion,
            llm=self.llm,
            feeder=self.feeder,
            dry_feeder=FakeFeeder(),
            notifier=self.notifier,
            clock=self.clock,
            frames_dir=self.tmp_path / "frames",
            dashboard_url="http://dash",
            capture_gate=self.capture_gate,
            capture_dir=self.tmp_path / "unlabeled",
            power=self.power,
        )

    def load_plates(self, *plates: int) -> None:
        """What marking bowls from the dashboard does: the marks are honoured as given."""
        for plate in plates:
            self.store.set_plate(plate, "loaded", self.clock.now)
        self.store.set("plates_marked", "1")

    def push_frame(self) -> None:
        self.frames.push(Frame(jpeg=TINY_JPEG, at=self.clock.now))

    def motion_now(self) -> None:
        self.motion.last_motion_at = self.clock.now

    def tick(self, *, motion: bool = True, seconds: float = 1.0) -> None:
        self.clock.advance(seconds=seconds)
        self.push_frame()
        if motion:
            self.motion_now()
        self.machine.tick()

    def run(self, *, seconds: float, motion: bool = True, step: float = 1.0) -> None:
        elapsed = 0.0
        while elapsed < seconds:
            self.tick(motion=motion, seconds=step)
            elapsed += step

    def state(self) -> str:
        return str(self.machine.state)

    def events(self, kind: str) -> list:
        return [e for e in self.store.events(limit=500) if e.kind == kind]


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)
