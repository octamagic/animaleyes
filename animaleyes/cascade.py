"""Cascade detector: a free local YOLO gate that only escalates to Claude when needed.

YOLO (local, $0) decides dog / cat / nothing on every check. We only pay for a Claude call
when a *dog* is actually at the bowl — the one case that needs real discrimination (Grrr vs
Bowie, and catching a cat that YOLO misread as a small dog). A cat or an empty scene is
handled locally for free. In practice Claude runs only while a dog is in view (a handful of
checks per visit) instead of on every motion, cutting cost while keeping Claude's accuracy.

The gate can be blind: with the camera tight on the feeder, a dog filling the frame (or half
out of it) often isn't recognised as a "dog" at all, and the visit used to go unanswered for
minutes. So when the gate sees nothing, Claude still gets a look every CASCADE_PROBE_S while
motion lasts, and once Claude has seen an animal the gate missed it keeps answering for
CASCADE_STICKY_S so the confirmation streak isn't broken by the gate's "nothing".

Every confident Claude dog verdict also saves the judged frame to data/training/<animal>/,
pre-labelled, so a classifier that eventually replaces Claude trains itself from real IR frames.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from .camera import Frame
from .detect import BedVerdict, FeedingVerdict, Identifier, Verdict

log = logging.getLogger(__name__)


class CascadeIdentifier:
    def __init__(
        self,
        gate: Identifier,
        confirm: Identifier,
        config=None,
        training_dir: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.gate = gate  # local YOLO
        self.confirm = confirm  # Claude
        self.config = config
        self.training_dir = training_dir
        self.monotonic = monotonic
        self.reference_counts: dict[str, int] = getattr(confirm, "reference_counts", {})
        self._last_saved = 0.0
        self._last_confirm_at: float | None = None
        self._last_confirm_saw_animal = False

    def identify(self, frames: list[Frame]) -> Verdict:
        v = self.gate.identify(frames)
        # Any dog (whoever the gate thinks it is) or an animal it can't name → Claude decides.
        # Only "cat" and "none" are settled locally: neither can lead to a feed.
        if v.animal not in ("none", "cat"):
            log.info("cascade: gate sees %s -> asking Claude who it is", v.animal)
            return self._confirm(frames)
        # The gate sees nothing, but identify() only runs while something is moving at the
        # bowl — so it may simply be blind. Let Claude look, on a throttle.
        if v.animal == "none" and self._probe_due():
            log.info("cascade: gate sees nothing during motion -> letting Claude look")
            return self._confirm(frames)
        # Cat, or nothing and not due for a probe → decided locally, no paid call.
        log.info("cascade: decided locally: %s (%s)", v.animal, v.reason)
        return v

    def _confirm(self, frames: list[Frame]) -> Verdict:
        v = self.confirm.identify(frames)
        self._last_confirm_at = self.monotonic()
        self._last_confirm_saw_animal = v.animal not in ("none", "unsure")
        self._maybe_collect(v, frames)
        return v

    def _probe_due(self) -> bool:
        if self.config is None:
            return False
        s = self.config.load()
        if s.CASCADE_PROBE_S <= 0:
            return False  # probing off: the gate alone decides (cheapest, can miss close-ups)
        if self._last_confirm_at is None:
            return True
        since = self.monotonic() - self._last_confirm_at
        if self._last_confirm_saw_animal and since <= s.CASCADE_STICKY_S:
            return True  # Claude just saw an animal the gate can't: stay with Claude
        return since >= s.CASCADE_PROBE_S

    def _maybe_collect(self, verdict: Verdict, frames: list[Frame]) -> None:
        if self.training_dir is None or not frames or self.config is None:
            return
        s = self.config.load()
        if not s.COLLECT_TRAINING or verdict.animal not in ("grrr", "bowie"):
            return
        if verdict.confidence < s.TRAINING_MIN_CONF:
            return
        now = time.monotonic()
        if now - self._last_saved < s.TRAINING_MIN_GAP_S:
            return
        self._last_saved = now
        try:
            frame = frames[-1]
            d = self.training_dir / verdict.animal
            d.mkdir(parents=True, exist_ok=True)
            name = f"{frame.at.strftime('%Y%m%d-%H%M%S')}_{int(verdict.confidence * 100)}.jpg"
            (d / name).write_bytes(frame.jpeg)
        except OSError as exc:
            log.warning("could not save training frame: %s", exc)

    def feeding_check(self, frames: list[Frame]) -> FeedingVerdict:
        # If the gate plainly sees Grrr, that is enough and free. Otherwise it may be blind
        # (frame-filling close-up, dog at the frame edge), and "absent" closes the lid on her —
        # so Claude decides, which also restores bowl state and other-animal detection while
        # the lid is open, the period that matters most.
        v = self.gate.feeding_check(frames)
        if v.grrr_at_bowl:
            return v
        return self.confirm.feeding_check(frames)

    def verify_food(self, frames: list[Frame]) -> FeedingVerdict:
        # Judging whether the served bowl has food needs real vision — route it to Claude
        # (YOLO can't see food). This runs only while a plate is being verified, so it's rare.
        return self.confirm.verify_food(frames)

    def bed_check(self, frames: list[Frame], empty_reference: bytes | None) -> BedVerdict:
        # The local gate cannot see her in the bed (see YoloIdentifier.bed_check); Claude can.
        return self.confirm.bed_check(frames, empty_reference)

    def reload_references(self) -> None:
        if hasattr(self.confirm, "reload_references"):
            self.confirm.reload_references()
            self.reference_counts = getattr(self.confirm, "reference_counts", {})
