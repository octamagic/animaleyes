"""The feeding state machine. This is the only code allowed to open or close the feeder.

States: OUTSIDE_WINDOW, IDLE, WATCHING, OPENING, FEEDING, CLOSING, COOLDOWN, DONE.
Every transition is written to the events table with a reason. Anything a restart must
remember (state, open timestamps, counters, plates) is persisted in the Store before the
side effect that depends on it, so a crash between "send feed" and "record feed" resumes as
FEEDING rather than opening again.

All time comes from the injected clock and all outside contact goes through injected
protocols, so the whole thing runs under test with fakes.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .camera import Frame, FrameBuffer
from .config import ConfigStore, Settings
from .feeder import Feeder, FeederError
from .motion import MotionDetector
from .notify import Notifier
from .store import Store
from .vision import FeedingVerdict, Identifier, Verdict

log = logging.getLogger(__name__)

CAMERA_OFFLINE_S = 60
VETO_REPEAT_S = 300
NONE_LEFT_REPEAT_S = 3600
VERDICT_HISTORY = 6
# One read this sure of a non-target animal is logged as a sighting on its own; below it we
# wait for the same animal twice in a row. Logging only — it never gates or triggers a feed.
SIGHTING_MIN_CONF = 0.6
NOT_AN_ANIMAL = ("grrr", "none", "unsure")
# States where the machine already runs its own identification or the lid is open.
WATCH_SKIP_STATES = ("WATCHING", "OPENING", "VERIFYING", "FEEDING", "CLOSING")


class State(StrEnum):
    OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
    IDLE = "IDLE"
    WATCHING = "WATCHING"
    OPENING = "OPENING"
    VERIFYING = "VERIFYING"
    FEEDING = "FEEDING"
    CLOSING = "CLOSING"
    COOLDOWN = "COOLDOWN"
    DONE = "DONE"


class Machine:
    def __init__(
        self,
        config: ConfigStore,
        store: Store,
        frames: FrameBuffer,
        motion: MotionDetector,
        llm: Identifier,
        feeder: Feeder,
        dry_feeder: Feeder,
        notifier: Notifier,
        clock: Callable[[], datetime],
        frames_dir: Path,
        dashboard_url: str = "",
        events=None,
        capture_gate: Identifier | None = None,
        capture_dir: Path | None = None,
        healthcheck: Callable[[], None] | None = None,
    ):
        self.config = config
        self.store = store
        self.frames = frames
        self.motion = motion
        self.events = events
        self.llm = llm
        self.real_feeder = feeder
        self.dry_feeder = dry_feeder
        self.notifier = notifier
        self.clock = clock
        self.frames_dir = frames_dir
        self.dashboard_url = dashboard_url
        # Dead-man's-switch ping (injected, best-effort). Fired each heartbeat so an external
        # monitor alerts if the loop stops, the process dies, or the box/network goes down.
        self.healthcheck = healthcheck
        # Capture mode: a free local detector + the unlabelled-frame directory. Both may be
        # None (no local detector available), in which case capture mode quietly does nothing.
        self.capture_gate = capture_gate
        self.capture_dir = capture_dir
        self.verdicts: deque[Verdict] = deque(maxlen=VERDICT_HISTORY)
        # When each confident Grrr read happened. Confirmation counts these within
        # CONFIRM_WINDOW_S; a "none"/"unsure" read in between is a bad frame, not evidence
        # she left, so it neither counts nor resets. Any other animal resets.
        self.grrr_seen: deque[datetime] = deque(maxlen=VERDICT_HISTORY)
        self.last_verdict: Verdict | FeedingVerdict | None = None
        self.last_verdict_at: datetime | None = None
        self.watch_verdicts: deque[Verdict] = deque(maxlen=2)
        self.last_watch_at: datetime | None = None
        self.judged_frames: list[Frame] = []
        self.last_llm_at: datetime | None = None
        self.last_feeding_verdict: FeedingVerdict | None = None
        self.last_heartbeat_at: datetime | None = None
        self.camera_offline = False
        self.done_motion_seen = False
        self.started_at = clock()
        self.settings: Settings = config.load()

    # persisted state ---------------------------------------------------
    @property
    def state(self) -> State:
        return State(self.store.get("state", State.IDLE))

    def _when(self, key: str) -> datetime | None:
        raw = self.store.get(key)
        return datetime.fromisoformat(raw) if raw else None

    def _transition(self, new_state: State, reason: str, **data: Any) -> None:
        old = self.state
        if old == new_state:
            return
        log.info("%s -> %s: %s %s", old, new_state, reason, data or "")
        self.store.set("state", new_state)
        self.store.add_event(
            self.clock(), "transition", reason, {"from": old, "to": new_state, **data}
        )

    def _feeder(self) -> Feeder:
        return self.dry_feeder if self.settings.DRY_RUN else self.real_feeder

    # startup -----------------------------------------------------------
    def start(self) -> None:
        self.settings = self.config.load()
        now = self.clock()
        state = self.state
        if state in (State.OPENING, State.VERIFYING, State.FEEDING):
            # The feed command may already have gone out. Assume the lid is open; never re-send
            # (and never re-verify/re-rotate on resume — just feed whatever is already served).
            self.store.set("state", State.FEEDING)
            self.store.set("grrr_last_seen_at", now.isoformat())
            self._mark_fed(now)  # assume food went out: the gap must apply after a restart
            resumed = "resumed FEEDING after restart; lid assumed open"
        elif state == State.CLOSING:
            resumed = "resumed CLOSING after restart"
        else:
            resumed = f"resumed {state}"
        event_id = self.store.add_event(now, "startup", resumed, self._status_data())
        self._notify(f"animaleyes started: {resumed}", event_id)

    # main loop ---------------------------------------------------------
    def tick(self) -> None:
        self.settings = self.config.load()
        self.motion.threshold = self.settings.MOTION_PIXEL_FRACTION
        now = self.clock()
        self._check_camera(now)
        self._heartbeat(now)
        self._poll_lid(now)
        self._consume_dashboard_requests(now)
        if self.settings.CAPTURE_MODE:
            self._tick_capture(now)
        self._tick_watch(now)

        state = self.state
        lid_open = state in (State.VERIFYING, State.FEEDING, State.CLOSING)
        if not self.settings.ENABLED and not lid_open:
            self._transition(State.OUTSIDE_WINDOW, "ENABLED is false")
            return
        if not self.settings.is_active_at(now.time()) and not lid_open:
            self._transition(State.OUTSIDE_WINDOW, "outside active window")
            return

        handler = {
            State.OUTSIDE_WINDOW: self._tick_outside_window,
            State.IDLE: self._tick_idle,
            State.WATCHING: self._tick_watching,
            State.OPENING: self._tick_opening,
            State.VERIFYING: self._tick_verifying,
            State.FEEDING: self._tick_feeding,
            State.CLOSING: self._tick_closing,
            State.COOLDOWN: self._tick_cooldown,
            State.DONE: self._tick_done,
        }[state]
        handler(now)

    def _tick_outside_window(self, now: datetime) -> None:
        self.store.set("feeds_this_window", 0)
        self.store.set("plates_tried", "[]")  # a new window: every bowl may have been refilled
        # Dashboard marks count for the window they were made in, or while a bowl is still
        # marked loaded. Nothing marked = the human may have refilled without telling us.
        self.store.set("plates_marked", "1" if self.store.loaded_plates() else None)
        self._forget_verdicts()
        self._transition(State.IDLE, "active window started")

    def _tick_idle(self, now: datetime) -> None:
        if not self._candidate_plates():
            self._transition(State.DONE, self._no_plates_reason())
            return
        if self._motion(now):
            self._forget_verdicts()
            self._transition(State.WATCHING, "motion detected", fraction=self.motion.last_fraction)

    def _tick_watching(self, now: datetime) -> None:
        if not self._motion(now):
            self._transition(State.IDLE, f"no motion for {self.settings.MOTION_HOLD_S}s")
            return
        if not self._llm_due(now, self.settings.LLM_MIN_INTERVAL_S):
            return
        self.judged_frames = self.frames.latest(3)
        verdict = self.llm.identify(self.judged_frames)
        self.last_llm_at = now
        self.last_verdict = verdict
        self.last_verdict_at = now
        self.verdicts.append(verdict)
        self._note_grrr(now, verdict)
        self._check_veto(now)
        if self._grrr_confirmed(now):
            blocker = self._open_blocker(now)
            if blocker:
                self._log_once(now, "grrr_blocked", blocker, VETO_REPEAT_S)
                return
            self._transition(State.OPENING, "grrr confirmed", verdict=verdict.to_dict())
            self._tick_opening(now, trigger="grrr confirmed")

    def _tick_opening(self, now: datetime, trigger: str = "resumed") -> None:
        feeder = self._feeder()
        mode = "DRY_RUN" if self.settings.DRY_RUN else "LIVE"
        frames = self._save_frames("open")
        try:
            plate = self._plate_under_lid(feeder, self._candidate_plates())
            self.store.set("opening_plate", plate)
            self.store.set("current_plate", plate)
            self._mark_tried(plate)  # persisted before the open, so a crash can't retry it
            feeder.open_now(plate)
        except FeederError as exc:
            # Back off so a failing feeder isn't hammered every tick (which drains the device
            # and spams the cloud). The dog's return after the window still re-triggers later.
            self.store.set("feed_failed_at", now.isoformat())
            event_id = self.store.add_event(
                now, "feed_failed", str(exc), {"trigger": trigger, "mode": mode}, frames
            )
            self._notify(f"FEED FAILED ({mode}): {exc}", event_id, urgent=True)
            self._transition(State.IDLE, "feed failed", error=str(exc))
            return
        self.store.set("feed_failed_at", None)
        self.store.set("lid_actual_open", 1)
        loaded = self.store.loaded_plates()
        self.store.set("feeding_plate", plate)
        self.store.set("opened_at", now.isoformat())
        self.store.set("last_open_at", now.isoformat())
        self.store.set("grrr_last_seen_at", now.isoformat())
        self.store.set("feeds_this_window", self.store.get_int("feeds_this_window") + 1)
        # Reset the verify counters for this feeding session (one initial open, no rotations yet).
        self.store.set("verify_rotations", 0)
        self.store.set("verify_empty_count", 0)
        self.store.set("verify_started_at", now.isoformat())
        self.last_feeding_verdict = None
        self.last_llm_at = now
        event_id = self.store.add_event(
            now,
            "open",
            trigger,
            {
                "plate": plate,
                "mode": mode,
                "loaded_plates": loaded,
                "plates_tried": self.plates_tried(),
                "feeds_this_window": self.store.get_int("feeds_this_window"),
            },
            frames,
        )
        marked = f"Loaded plates: {loaded}" if loaded else "no bowl was marked loaded; trying it"
        self._notify(f"OPENED plate {plate} ({mode}): {trigger}. {marked}", event_id)
        if self.settings.VERIFY_FOOD:
            self._transition(State.VERIFYING, "lid open, verifying food", plate=plate)
        else:
            self._mark_fed(now)
            self._transition(State.FEEDING, "lid open", plate=plate)

    def _tick_verifying(self, now: datetime) -> None:
        """Lid is open; confirm the served bowl actually has food before settling into FEEDING.

        food -> FEEDING. Confirmed empty (VERIFY_EMPTY_CONFIRMATIONS in a row) -> close, rotate to
        the next loaded plate and re-open, up to MAX_ROTATE_FOR_FOOD; out of plates/rotations ->
        alert and CLOSING. unsure never rotates, and after VERIFY_TIMEOUT_S of not knowing we stop
        second-guessing and feed, so a bad read can never starve Grrr or churn the whole tray.
        """
        opened_at = self._when("opened_at") or now
        if now - opened_at >= timedelta(minutes=self.settings.FEEDING_MAX_MIN):
            self._transition(State.CLOSING, "FEEDING_MAX_MIN reached while verifying")
            self._tick_closing(now)
            return
        if not self._llm_due(now, self.settings.VERIFY_POLL_S):
            return
        self.judged_frames = self.frames.latest(3)
        verdict = self.llm.verify_food(self.judged_frames)
        self.last_llm_at = now
        self.last_verdict = verdict
        self.last_verdict_at = now
        self.last_feeding_verdict = verdict
        plate = self.store.get_int("feeding_plate", 0)
        if verdict.grrr_at_bowl:
            # She is eating while we check the plate: that is presence, so the leave timer
            # must not run from the moment the lid opened.
            self.store.set("grrr_last_seen_at", now.isoformat())

        if verdict.bowl == "food":
            self.store.set("verify_empty_count", 0)
            self._mark_fed(now)
            self._transition(State.FEEDING, f"food confirmed on plate {plate}")
            return

        if verdict.bowl != "empty":  # unsure
            self.store.set("verify_empty_count", 0)
            started = self._when("verify_started_at") or opened_at
            if now - started >= timedelta(seconds=self.settings.VERIFY_TIMEOUT_S):
                self._mark_fed(now)
                self._transition(State.FEEDING, "food unverified (timed out), feeding anyway")
            return

        # Confirmed-ish empty: require consecutive reads so one misjudged frame can't rotate.
        count = self.store.get_int("verify_empty_count") + 1
        self.store.set("verify_empty_count", count)
        if count < self.settings.VERIFY_EMPTY_CONFIRMATIONS:
            return
        self._handle_empty_plate(now, plate)

    def _handle_empty_plate(self, now: datetime, plate: int) -> None:
        rotations = self.store.get_int("verify_rotations")
        # The served plate is empty — record it so it's not offered again (updates the dashboard).
        if plate:
            self.store.set_plate(plate, "empty", now)
        others = [p for p in self._candidate_plates() if p != plate]
        frames = self._save_frames("empty_plate")
        if rotations >= self.settings.MAX_ROTATE_FOR_FOOD or not others:
            reason = (
                "served plate empty; rotation cap reached"
                if rotations >= self.settings.MAX_ROTATE_FOR_FOOD
                else "served plate empty; no other plate left to try"
            )
            event_id = self.store.add_event(
                now, "empty_no_food", reason, {"plate": plate, "rotations": rotations}, frames
            )
            self._notify(
                f"EMPTY PLATE: {reason}. Closing. Check the feeder.", event_id, urgent=True
            )
            self._transition(State.CLOSING, reason)
            self._tick_closing(now)
            return
        try:
            new_plate = self._reserve_next_loaded(now)
        except FeederError as exc:
            # The device refused mid-swap (motor error, timeout). Back off like a failed open so
            # it isn't hammered, but don't treat this as a feed: nothing was served.
            self.store.set("feed_failed_at", now.isoformat())
            event_id = self.store.add_event(
                now, "empty_no_food", f"re-serve failed: {exc}", {"plate": plate}, frames
            )
            self._notify(f"EMPTY PLATE and re-serve FAILED: {exc}. Closing.", event_id, urgent=True)
            self._transition(State.CLOSING, "re-serve failed")
            self._tick_closing(now)
            return
        self.store.set("verify_empty_count", 0)
        self.store.set("verify_started_at", now.isoformat())
        self.store.set("verify_rotations", rotations + 1)
        event_id = self.store.add_event(
            now,
            "rotated_empty_plate",
            f"plate {plate} empty -> rotated to plate {new_plate}",
            {"from": plate, "to": new_plate, "rotations": rotations + 1},
            frames,
        )
        self._notify(
            f"Plate {plate} was empty; rotated to plate {new_plate} and re-opened.", event_id
        )

    def _reserve_next_loaded(self, now: datetime) -> int:
        """Close the empty plate, rotate to the next still-loaded plate, and open it. The lid is
        open on an empty plate when this is called; closing first keeps the tray movement safe."""
        feeder = self._feeder()
        feeder.close()
        self.store.set("lid_actual_open", 0)
        plate = self._plate_under_lid(feeder, self._candidate_plates())  # rotates while closed
        self._mark_tried(plate)
        feeder.open_now(plate)
        self.store.set("lid_actual_open", 1)
        self.store.set("current_plate", plate)
        self.store.set("feeding_plate", plate)
        # Reset the feeding clock to this plate; this is a correction within the same session, so
        # last_open_at and feeds_this_window are deliberately left untouched.
        self.store.set("opened_at", now.isoformat())
        self.store.set("grrr_last_seen_at", now.isoformat())
        return plate

    def _tick_feeding(self, now: datetime) -> None:
        opened_at = self._when("opened_at") or now
        last_seen = self._when("grrr_last_seen_at") or opened_at
        if now - opened_at >= timedelta(minutes=self.settings.FEEDING_MAX_MIN):
            self._transition(
                State.CLOSING, f"FEEDING_MAX_MIN ({self.settings.FEEDING_MAX_MIN}) reached"
            )
            self._tick_closing(now)
            return
        if now - last_seen >= timedelta(seconds=self.settings.LEAVE_TIMEOUT_S):
            self._transition(State.CLOSING, f"grrr absent for {self.settings.LEAVE_TIMEOUT_S}s")
            self._tick_closing(now)
            return
        if not self._llm_due(now, self.settings.FEEDING_POLL_S):
            return
        self.judged_frames = self.frames.latest(3)
        verdict = self.llm.feeding_check(self.judged_frames)
        self.last_llm_at = now
        self.last_verdict = verdict
        self.last_verdict_at = now
        self.last_feeding_verdict = verdict
        if verdict.grrr_at_bowl:
            self.store.set("grrr_last_seen_at", now.isoformat())
        for animal in verdict.other_animals_present:
            if animal not in ("grrr", "none", "unsure"):  # any non-target real animal
                self._log_once(
                    now,
                    f"{animal}_during_feed",
                    verdict.reason,
                    VETO_REPEAT_S,
                    frames=self._save_frames(f"{animal}_during_feed"),
                    notify=True,
                )

    def _tick_closing(self, now: datetime) -> None:
        feeder = self._feeder()
        plate = self.store.get_int("feeding_plate", 0)
        opened_at = self._when("opened_at") or now
        bowl = self.last_feeding_verdict.bowl if self.last_feeding_verdict else "unsure"
        # "empty" or "unsure" advances to the next plate next time; only a clearly still-full
        # bowl is offered again. Wrongly advancing costs one bowl; wrongly staying starves.
        plate_status = "loaded" if bowl == "food" else "eaten"
        error = None
        close_sent = True
        try:
            close_sent = feeder.close() is not False
        except FeederError as exc:
            error = str(exc)
        if plate:
            self.store.set_plate(plate, plate_status, now)
        session_s = int((now - opened_at).total_seconds())
        last_fed = self._when("last_fed_at")
        fed = bool(last_fed and last_fed >= opened_at)
        data = {
            "plate": plate,
            "bowl": bowl,
            "plate_status": plate_status,
            "session_s": session_s,
            "fed": fed,
            "loaded_plates": self.store.loaded_plates(),
            "mode": "DRY_RUN" if self.settings.DRY_RUN else "LIVE",
            # False = the feeder said no lid was open, so nothing was sent. Usually it closed
            # itself; if the lid is in fact open, the feeder and its cloud are out of sync.
            "close_sent": close_sent,
        }
        frames = self._save_frames("close")
        if error:
            event_id = self.store.add_event(now, "close_failed", error, data, frames)
            self._notify(f"CLOSE FAILED after {session_s}s: {error}", event_id, urgent=True)
        else:
            how = "" if close_sent else " (feeder reported the lid already closed; nothing sent)"
            event_id = self.store.add_event(now, "close", f"bowl {bowl}{how}", data, frames)
            self._notify(
                f"CLOSED plate {plate} after {session_s}s, bowl {bowl} -> {plate_status}{how}. "
                f"Loaded plates left: {self.store.loaded_plates()}",
                event_id,
            )
        self.store.set("lid_actual_open", 0)
        self._forget_verdicts()
        if not fed:
            # Nothing was served (empty bowl, swap refused): the gap guards against a second
            # helping, and there was no first. Let her try the next bowl as soon as she's back.
            self._transition(State.IDLE, "lid closed, no food served: no cooldown", **data)
            return
        self._transition(State.COOLDOWN, "lid closed", **data)

    def _tick_cooldown(self, now: datetime) -> None:
        last_fed = self._when("last_fed_at")
        if last_fed and now - last_fed < timedelta(minutes=self.settings.MIN_GAP_MIN):
            return
        if self._candidate_plates():
            self._transition(State.IDLE, f"MIN_GAP_MIN ({self.settings.MIN_GAP_MIN}) elapsed")
        else:
            self._transition(State.DONE, self._no_plates_reason())

    def _tick_done(self, now: datetime) -> None:
        if self._candidate_plates():
            self._transition(State.IDLE, "plates available")
            return
        motion = self._motion(now)
        if not motion:
            self.done_motion_seen = False
            return
        if self.done_motion_seen:
            return
        self.done_motion_seen = True
        self.judged_frames = self.frames.latest(3)
        verdict = self.llm.identify(self.judged_frames)
        self.last_llm_at = now
        self.last_verdict = verdict
        self.last_verdict_at = now
        if verdict.animal == "grrr":
            self._log_once(
                now,
                "wanted_food_none_left",
                verdict.reason,
                NONE_LEFT_REPEAT_S,
                frames=self._save_frames("none_left"),
                notify=True,
            )
        self._log_sightings(now, "sighting", [verdict])

    # helpers -----------------------------------------------------------
    def _motion(self, now: datetime) -> bool:
        hold_s = self.settings.MOTION_HOLD_S
        # Prefer the camera's own motion/animal events (cheap, selective) so the LLM only
        # wakes when the camera sees something. Fall back to frame-differencing if the event
        # source is unavailable or its subscription is currently failing, so a broken ONVIF
        # link can never silently stop the dog from being fed.
        if (
            self.events is not None
            and self.settings.MOTION_SOURCE == "camera"
            and getattr(self.events, "last_error", "unset") is None
        ):
            return self.events.motion_within(now, hold_s)
        return self.motion.motion_within(now, hold_s)

    def _llm_due(self, now: datetime, interval_s: float) -> bool:
        return self.last_llm_at is None or now - self.last_llm_at >= timedelta(seconds=interval_s)

    def _note_grrr(self, now: datetime, verdict: Verdict) -> None:
        """Feed one identify() result into the confirmation count."""
        if verdict.is_grrr(self.settings.GRRR_MIN_CONF, self.settings.OPEN_REQUIRES_AT_BOWL):
            self.grrr_seen.append(now)
        elif verdict.animal not in ("grrr", "none", "unsure") or verdict.other_animals_present:
            # A different animal seen (alone or beside her) contradicts the streak. A shaky
            # Grrr read, nothing, or unsure is just an unhelpful frame: neither counts nor resets.
            self.grrr_seen.clear()

    def _grrr_confirmed(self, now: datetime) -> bool:
        """CONFIRMATIONS_REQUIRED confident Grrr reads within CONFIRM_WINDOW_S. With the camera
        tight on the feeder she bobs in and out of frame while she waits, so demanding the reads
        be back to back used to take ten minutes of her scratching at the lid."""
        window = timedelta(seconds=self.settings.CONFIRM_WINDOW_S)
        recent = [t for t in self.grrr_seen if now - t <= window]
        return len(recent) >= self.settings.CONFIRMATIONS_REQUIRED

    def _forget_verdicts(self) -> None:
        self.verdicts.clear()
        self.grrr_seen.clear()

    def _mark_fed(self, now: datetime) -> None:
        """Food is (assumed) in front of her: this is what MIN_GAP_MIN counts from. An open that
        served nothing (empty bowl, refused swap) does not move it."""
        self.store.set("last_fed_at", now.isoformat())

    def _open_blocker(self, now: datetime) -> str | None:
        if not self._candidate_plates():
            return self._no_plates_reason()
        last_fed = self._when("last_fed_at")
        if last_fed and now - last_fed < timedelta(minutes=self.settings.MIN_GAP_MIN):
            return f"last fed {int((now - last_fed).total_seconds() // 60)} min ago < MIN_GAP_MIN"
        failed = self._when("feed_failed_at")
        if failed and now - failed < timedelta(seconds=self.settings.FEED_RETRY_BACKOFF_S):
            return f"feed failed {int((now - failed).total_seconds())}s ago < FEED_RETRY_BACKOFF_S"
        return None

    def _check_veto(self, now: datetime) -> None:
        self._log_sightings(now, "veto", list(self.verdicts)[-2:])

    def _log_sightings(
        self,
        now: datetime,
        kind: str,
        recent: list[Verdict],
        *,
        frames: list[Frame] | None = None,
        include_target: bool = False,
    ) -> None:
        """Record every non-target animal in the latest verdict, once per animal per
        VETO_REPEAT_S. A sighting needs one confident read, or the same animal twice in a row,
        so a single shaky frame can't cry cat. Animals named in other_animals_present count too
        (Chicken next to Grrr is still Chicken at the bowl).

        Deduped per animal, not per kind: Bowie wandering past must not silence the cat."""
        if not recent:
            return
        latest = recent[-1]
        skip = ("none", "unsure") if include_target else NOT_AN_ANIMAL
        steady = len(recent) >= 2 and recent[-2].animal == latest.animal
        seen = []
        if latest.animal not in skip and (steady or latest.confidence >= SIGHTING_MIN_CONF):
            seen.append(latest.animal)
        seen += [a for a in latest.other_animals_present if a not in skip and a not in seen]
        for animal in seen:
            self._log_once(
                now,
                kind,
                f"{animal}: {latest.reason}",
                VETO_REPEAT_S,
                data={"animal": animal, "verdict": latest.to_dict()},
                frames=lambda tag=f"{kind}_{animal}": self._save_frames(tag, frames),
                notify=True,
                dedupe=f"{kind}:{animal}",
            )

    # plates -------------------------------------------------------------
    def plates_tried(self) -> list[int]:
        """Bowls opened so far this window (persisted; reset when a window starts)."""
        try:
            return [int(p) for p in json.loads(self.store.get("plates_tried") or "[]")]
        except (ValueError, TypeError):
            return []

    def _mark_tried(self, plate: int) -> None:
        tried = self.plates_tried()
        if plate not in tried:
            self.store.set("plates_tried", json.dumps(tried + [plate]))

    def _candidate_plates(self) -> list[int]:
        """Bowls the machine may open now. Marked-loaded bowls win, exactly as marked. With nothing
        marked for this window (the human may have refilled without saying so), every bowl not yet
        tried this window is a candidate, so a forgotten tick box can't mean a hungry dog."""
        loaded = self.store.loaded_plates()
        if loaded:
            return loaded
        if self.store.get("plates_marked") or not self.settings.FEED_WITHOUT_LOADED_PLATES:
            return []
        tried = self.plates_tried()
        return [p for p in (1, 2, 3) if p not in tried]

    def _no_plates_reason(self) -> str:
        if self.store.get("plates_marked") or not self.settings.FEED_WITHOUT_LOADED_PLATES:
            return "no loaded plates"
        return "every bowl tried this window"

    def _plate_under_lid(self, feeder: Feeder, candidates: list[int]) -> int:
        """Rotate until one of `candidates` is under the lid. No candidates = any plate (manual)."""
        plate = feeder.current_plate()
        if not candidates:
            return plate
        for _ in range(3):
            if plate in candidates:
                return plate
            feeder.rotate()
            plate = feeder.current_plate()
        raise FeederError(
            f"no candidate plate reachable; tray reports plate {plate}, want {candidates}"
        )

    def _log_once(
        self,
        now: datetime,
        kind: str,
        reason: str,
        repeat_s: int,
        *,
        data: dict[str, Any] | None = None,
        frames: list[str] | Callable[[], list[str]] | None = None,
        notify: bool = False,
        dedupe: str | None = None,
    ) -> None:
        # `dedupe` narrows the repeat window below the event kind (e.g. per animal); persisted
        # so a restart doesn't re-announce what was just announced.
        if dedupe:
            last = self._when(f"logged_at:{dedupe}")
        else:
            last = self.store.last_event_at(kind)
        if last and now - last < timedelta(seconds=repeat_s):
            return
        if dedupe:
            self.store.set(f"logged_at:{dedupe}", now.isoformat())
        if callable(frames):  # only write images for an event that is actually logged
            frames = frames()
        event_id = self.store.add_event(now, kind, reason, data, frames)
        if notify:
            self._notify(f"{kind}: {reason}", event_id)

    def _tick_watch(self, now: datetime) -> None:
        """Always-on local look at the bowl, so the log and the dashboard know who's there even
        when the feeding machine isn't looking: off hours, disabled, idle, cooldown, done.

        Uses only the free local detector, every WATCH_INTERVAL_S, motion or not (a cat that
        sits and stares makes no motion). Visibility only: it updates the shown verdict and
        logs per-animal sightings. It never moves the machine or the feeder, and Grrr seen
        here still has to be confirmed in WATCHING before anything opens.
        """
        if not self.settings.ALWAYS_WATCH or self.capture_gate is None:
            return
        if self.state in WATCH_SKIP_STATES:
            return  # the machine is already identifying (or the lid is open)
        last = self.last_watch_at
        if last and (now - last).total_seconds() < max(1, self.settings.WATCH_INTERVAL_S):
            return
        frames = self.frames.latest(1)
        if not frames:
            return
        self.last_watch_at = now  # in memory: a restart re-checking sooner is harmless
        try:
            verdict = self.capture_gate.identify(frames)
        except Exception as exc:  # noqa: BLE001 - a broken detector must not stall the loop
            self._log_once(now, "watch_error", f"detector failed: {exc}", 3600)
            return
        self.last_verdict = verdict
        self.last_verdict_at = now
        self.watch_verdicts.append(verdict)
        self._log_sightings(
            now, "sighting", list(self.watch_verdicts), frames=frames, include_target=True
        )

    def _tick_capture(self, now: datetime) -> None:
        """Capture mode: on any motion the local detector reads as an animal, save one frame,
        unlabelled, to the capture directory for the human to tag later for YOLO training.

        Runs independently of the feeding state machine — regardless of the active window or
        ENABLED — and only ever writes image files, so it can never move the feeder.
        """
        if self.capture_gate is None or self.capture_dir is None:
            return
        if not self._motion(now):
            return
        last = self._when("capture_last_at")
        gap = max(1, self.settings.CAPTURE_MIN_GAP_S)
        if last and (now - last).total_seconds() < gap:
            return
        frames = self.frames.latest(1)
        if not frames:
            return
        try:
            verdict = self.capture_gate.identify(frames)
        except Exception as exc:  # noqa: BLE001 - a broken detector must not stall the loop
            self._log_once(now, "capture_error", f"detector failed: {exc}", 3600)
            return
        if verdict.animal in ("none", "unsure"):
            return  # motion, but no animal the detector recognised — don't save noise
        self.store.set("capture_last_at", now.isoformat())
        frame = frames[-1]
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        # Filename carries the detector's rough guess (a hint for tagging) but the frame goes to
        # the unlabelled bucket; the human assigns the real label.
        stamp = frame.at.strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{verdict.animal}_{int(verdict.confidence * 100)}.jpg"
        try:
            (self.capture_dir / name).write_bytes(frame.jpeg)
        except OSError as exc:
            self._log_once(now, "capture_error", f"write failed: {exc}", 3600)
            return
        self.store.set("capture_count", self.store.get_int("capture_count") + 1)

    def _save_frames(self, tag: str, frames: list[Frame] | None = None) -> list[str]:
        # Default to the exact frames the detector just judged (self.judged_frames) so an
        # event's images show what the decision was actually made on — the LLM call takes a
        # second or two, during which a passing cat can leave frame, so grabbing fresh frames
        # here would save an empty scene. Falls back to the latest frames when none were judged.
        frames = frames if frames is not None else (self.judged_frames or self.frames.latest(3))
        names: list[str] = []
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            name = f"{frame.at.strftime('%Y%m%d-%H%M%S')}_{tag}_{i}.jpg"
            (self.frames_dir / name).write_bytes(frame.jpeg)
            names.append(name)
        return names

    def _notify(self, text: str, event_id: int | None = None, urgent: bool = False) -> None:
        # No credentials in the link; the browser asks for the dashboard's basic auth.
        if event_id and self.dashboard_url:
            text = f"{text}\n{self.dashboard_url}/events/{event_id}"
        # urgent -> email/SMS too; routine -> Slack/log only (keeps the phone quiet).
        self.notifier.alert(text) if urgent else self.notifier.send(text)

    def _check_camera(self, now: datetime) -> None:
        last = self.frames.last_frame_at() or self.started_at
        offline = now - last > timedelta(seconds=CAMERA_OFFLINE_S)
        if offline and not self.camera_offline:
            self.camera_offline = True
            event_id = self.store.add_event(
                now, "camera_offline", f"no frame for {CAMERA_OFFLINE_S}s"
            )
            self._notify("CAMERA OFFLINE: no frames", event_id, urgent=True)
        elif not offline and self.camera_offline:
            self.camera_offline = False
            event_id = self.store.add_event(now, "camera_online", "frames resumed")
            self._notify("camera back online", event_id)

    def _heartbeat(self, now: datetime) -> None:
        interval = timedelta(minutes=self.settings.HEARTBEAT_MIN)
        if self.last_heartbeat_at and now - self.last_heartbeat_at < interval:
            return
        self.last_heartbeat_at = now
        self.store.set("heartbeat_at", now.isoformat())
        status = self._status_data()
        self.notifier.send(
            f"♥ {status['state']} plates={status['loaded_plates']} "
            f"feeds={status['feeds_this_window']} "
            f"camera={'OFFLINE' if self.camera_offline else 'ok'}"
        )
        # Tell the external dead-man's-switch we're alive. Best-effort; if the box, loop, or
        # network is down these pings stop and the external monitor is what alerts the human.
        if self.healthcheck:
            try:
                self.healthcheck()
            except Exception as exc:  # noqa: BLE001 - monitoring must never affect feeding
                log.debug("healthcheck ping failed: %s", exc)

    def _poll_lid(self, now: datetime) -> None:
        """Read the feeder's real lid state so the dashboard reflects opens done out-of-band
        (e.g. from the PetLibro app), not just opens the machine made. Throttled, best-effort;
        uses _feeder() so DRY_RUN never touches the real device."""
        interval = self.settings.LID_POLL_S
        if interval <= 0:
            return
        last = self._when("lid_checked_at")
        if last and (now - last).total_seconds() < interval:
            return
        self.store.set("lid_checked_at", now.isoformat())
        try:
            self.store.set("lid_actual_open", 1 if self._feeder().manual_feed_active() else 0)
        except FeederError:
            pass  # leave the last known value rather than guessing

    def _consume_dashboard_requests(self, now: datetime) -> None:
        if self.store.get("plates_updated"):
            self.store.set("plates_updated", None)
            self.store.set("feeds_this_window", 0)
            self.store.set("plates_tried", "[]")
            self.store.set("plates_marked", "1")  # an explicit save: honour it as marked
            if self.state in (State.DONE, State.COOLDOWN, State.WATCHING, State.IDLE):
                self.store.add_event(
                    now,
                    "plates_set",
                    "plates set from dashboard",
                    {"loaded_plates": self.store.loaded_plates()},
                )
                self._transition(State.IDLE, "plates set from dashboard")
        # Manual feeder controls from the dashboard. Per the invariant, the page only sets
        # these flags; the machine is still the sole caller of the feeder. POC: these are
        # raw maintenance actions (open/close/rotate the tray by hand); they log an event and
        # honour DRY_RUN but deliberately do not touch the feeding counters or FSM state.
        for action in ("open", "close", "rotate"):
            if self.store.get(f"manual_{action}_requested"):
                self.store.set(f"manual_{action}_requested", None)
                self._manual_action(now, action)

        if self.store.get("feed_now_requested"):
            self.store.set("feed_now_requested", None)
            if self.state in (State.FEEDING, State.CLOSING, State.OPENING):
                self.store.add_event(now, "feed_now_ignored", "lid already open")
                return
            self._transition(State.OPENING, "manual feed from dashboard")
            self._tick_opening(now, trigger="manual feed from dashboard")

    def _manual_action(self, now: datetime, action: str) -> None:
        feeder = self._feeder()
        mode = "DRY_RUN" if self.settings.DRY_RUN else "LIVE"
        try:
            if action == "open":
                plate = self._plate_under_lid(feeder, [])
                feeder.open_now(plate)
                self.store.set("current_plate", plate)
                self.store.set("lid_manual_open", 1)
                self.store.set("lid_actual_open", 1)
                detail = {"mode": mode, "plate": plate}
            elif action == "close":
                if feeder.close() is False:
                    # Nothing was sent: the feeder reports no open lid. Say so instead of
                    # "closed" — the human pressed Close because they can see it open.
                    event_id = self.store.add_event(
                        now,
                        "manual_close_noop",
                        "feeder reports the lid is already closed; no close command was sent",
                        {"mode": mode},
                        self._save_frames("manual_close_noop", self.frames.latest(1)),
                    )
                    self._notify(
                        f"manual close NOT SENT ({mode}): feeder says already closed", event_id
                    )
                    return
                self.store.set("lid_manual_open", None)
                self.store.set("lid_actual_open", 0)
                detail = {"mode": mode}
            else:  # rotate
                feeder.rotate()
                plate = feeder.current_plate()
                self.store.set("current_plate", plate)
                detail = {"mode": mode, "plate": plate}
        except FeederError as exc:
            event_id = self.store.add_event(
                now, f"manual_{action}_failed", str(exc), {"mode": mode}
            )
            self._notify(f"manual {action} FAILED ({mode}): {exc}", event_id)
            return
        event_id = self.store.add_event(now, f"manual_{action}", f"dashboard ({mode})", detail)
        self._notify(f"manual {action} ({mode})", event_id)

    def request_feed_now(self) -> None:
        self.store.set("feed_now_requested", 1)

    def request_manual_action(self, action: str) -> None:
        """Enqueue a raw open/close/rotate for the machine to perform on its next tick."""
        if action not in ("open", "close", "rotate"):
            raise ValueError(f"unknown feeder action {action!r}")
        self.store.set(f"manual_{action}_requested", 1)

    def _status_data(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "loaded_plates": self.store.loaded_plates(),
            "plates": self.store.plates(),
            "plates_tried": self.plates_tried(),
            "feeds_this_window": self.store.get_int("feeds_this_window"),
            "last_open_at": self.store.get("last_open_at"),
            "opened_at": self.store.get("opened_at") if self.state == State.FEEDING else None,
            "feeding_plate": self.store.get("feeding_plate")
            if self.state == State.FEEDING
            else None,
        }

    def next_allowed_feed(self) -> datetime | None:
        last_fed = self._when("last_fed_at")
        if not last_fed:
            return None
        return last_fed + timedelta(minutes=self.settings.MIN_GAP_MIN)
