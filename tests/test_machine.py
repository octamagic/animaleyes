from __future__ import annotations

from datetime import datetime

from animaleyes.detect import FeedingVerdict, Verdict
from tests.conftest import (
    BOWIE,
    CAT,
    EATING,
    FOOD_UNSURE,
    GONE_EMPTY,
    GONE_FULL,
    GRRR,
    HAS_FOOD,
    NO_FOOD,
    UNSURE,
    FakeLLM,
    Harness,
)


def confirm_grrr(h: Harness) -> None:
    """Enough motion + Grrr verdicts to satisfy CONFIRMATIONS_REQUIRED at the LLM rate limit."""
    h.llm.identify_result = GRRR
    h.run(seconds=12)


def test_starts_idle_inside_window_and_watches_on_motion(h: Harness) -> None:
    h.load_plates(1)
    h.tick(motion=False)
    assert h.state() == "IDLE"
    h.tick(motion=True)
    assert h.state() == "WATCHING"


def test_outside_window_makes_no_llm_calls(tmp_path) -> None:
    h = Harness(tmp_path, start=datetime(2026, 9, 24, 12, 0))
    h.load_plates(1)
    h.llm.identify_result = GRRR
    h.run(seconds=30)
    assert h.state() == "OUTSIDE_WINDOW"
    assert h.llm.identify_calls == 0


def test_two_chamber_night(h: Harness) -> None:
    h.load_plates(1, 2)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1"]
    assert h.store.get_int("feeds_this_window") == 1

    # She eats, then leaves; the empty bowl is marked eaten and the lid closes.
    h.llm.feeding_result = EATING
    h.run(seconds=60)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=125)
    assert h.state() == "COOLDOWN"
    assert h.feeder.calls[-1] == "close"
    assert h.store.plates()[1] == "eaten"
    assert h.store.loaded_plates() == [2]

    # After MIN_GAP_MIN she comes back: the tray rotates to plate 2 and opens.
    h.clock.advance(minutes=61)
    h.tick(motion=False)
    assert h.state() == "IDLE"
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1", "open:2"]
    assert "rotate" in h.feeder.calls

    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=125)
    h.clock.advance(minutes=61)
    h.tick(motion=False)
    assert h.state() == "DONE"
    assert h.store.get_int("feeds_this_window") == 2
    assert [e.kind for e in h.store.events(kinds=("open", "close"))] == [
        "close",
        "open",
        "close",
        "open",
    ]


def test_min_gap_blocks_a_second_open(h: Harness) -> None:
    h.load_plates(1, 2)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=125)
    assert h.state() == "COOLDOWN"
    h.clock.advance(minutes=30)
    confirm_grrr(h)
    assert h.state() == "COOLDOWN"
    assert h.feeder.opens == ["open:1"]
    h.clock.advance(minutes=31)
    h.tick(motion=False)
    assert h.state() == "IDLE"
    confirm_grrr(h)
    assert h.feeder.opens == ["open:1", "open:2"]


def test_min_gap_applies_even_when_grrr_is_confirmed_in_watching(h: Harness) -> None:
    h.load_plates(1, 2)
    h.store.set("last_fed_at", h.clock.now.isoformat())
    confirm_grrr(h)
    assert h.state() == "WATCHING"
    assert h.feeder.opens == []
    assert h.events("grrr_blocked")


def test_leave_timeout_closes_and_presence_resets_it(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_FULL
    h.run(seconds=60)
    assert h.state() == "FEEDING"
    h.llm.feeding_result = EATING
    h.run(seconds=30)
    h.llm.feeding_result = GONE_FULL
    h.run(seconds=100)
    assert h.state() == "FEEDING", "presence must reset the leave timer"
    h.run(seconds=40)
    assert h.state() == "COOLDOWN"
    # She left food behind, so plate 1 is offered again next time.
    assert h.store.plates()[1] == "loaded"


def test_empty_bowl_and_away_closes_without_waiting_out_the_leave_timer(h: Harness) -> None:
    h.config.update({"LEAVE_TIMEOUT_S": 600})
    h.load_plates(1, 2)
    confirm_grrr(h)
    h.llm.feeding_result = EATING
    h.run(seconds=60)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=45)  # two polls at FEEDING_POLL_S=20, far short of 600s
    assert h.state() == "COOLDOWN"
    assert h.feeder.calls[-1] == "close"
    assert h.store.plates()[1] == "eaten"
    assert any("bowl empty and grrr away" in e.reason for e in h.events("transition"))


def test_licking_an_empty_bowl_keeps_the_lid_open(h: Harness) -> None:
    h.config.update({"LEAVE_TIMEOUT_S": 600})
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = FeedingVerdict(grrr_at_bowl=True, bowl="empty", reason="licking")
    h.run(seconds=120)
    assert h.state() == "FEEDING"


def test_one_empty_read_is_not_enough_and_unsure_resets_the_count(h: Harness) -> None:
    h.config.update({"LEAVE_TIMEOUT_S": 600})
    h.load_plates(1)
    confirm_grrr(h)
    for _ in range(3):
        h.llm.feeding_result = GONE_EMPTY
        h.run(seconds=20)
        h.llm.feeding_result = FOOD_UNSURE
        h.run(seconds=20)
    assert h.state() == "FEEDING"


def test_close_when_empty_off_waits_for_the_leave_timer(h: Harness) -> None:
    h.config.update({"LEAVE_TIMEOUT_S": 600, "CLOSE_WHEN_EMPTY": False})
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=300)
    assert h.state() == "FEEDING"
    h.run(seconds=310)
    assert h.state() == "COOLDOWN"


def test_feeding_max_closes_even_while_eating(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = EATING
    h.run(seconds=15 * 60 + 5, step=5)
    assert h.state() == "COOLDOWN"
    assert [e.reason for e in h.store.events(kinds=("transition",)) if e.data["to"] == "CLOSING"][
        0
    ].startswith("FEEDING_MAX_MIN")


def test_veto_during_feed_notifies_and_keeps_feeding(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = EATING.__class__(
        grrr_at_bowl=True, bowl="food", other_animals_present=["bowie"], reason="bowie hovering"
    )
    h.run(seconds=45)
    assert h.state() == "FEEDING"
    assert len(h.events("bowie_during_feed")) == 1
    assert any("bowie_during_feed" in s for s in h.notifier.sent)
    assert h.feeder.calls[-1].startswith("open")


def test_veto_in_watching_for_bowie_and_cat(h: Harness) -> None:
    h.load_plates(1)
    h.llm.identify_result = BOWIE
    h.run(seconds=8)
    assert h.state() == "WATCHING"
    assert [e.data["animal"] for e in h.events("veto")] == ["bowie"]
    # The cat right after Bowie still gets its own event; the repeat limit is per animal.
    h.llm.identify_result = CAT
    h.run(seconds=8)
    assert sorted(e.data["animal"] for e in h.events("veto")) == ["bowie", "cat"]
    h.run(seconds=60)
    assert len(h.events("veto")) == 2  # same cat lingering: rate-limited within VETO_REPEAT_S
    assert any("veto: cat" in s for s in h.notifier.sent)
    assert h.feeder.opens == []


def test_one_confident_cat_read_is_enough_to_log(h: Harness) -> None:
    # A cat that glances into the camera once, then an unsure frame, is still recorded.
    h.load_plates(1)
    h.tick()
    h.llm.identify_result = CAT
    h.tick()
    h.llm.identify_result = UNSURE
    h.run(seconds=10)
    assert len(h.events("veto")) == 1
    assert h.events("veto")[0].frames


def test_a_shaky_cat_read_needs_a_second_one(h: Harness) -> None:
    h.load_plates(1)
    h.tick()
    h.llm.identify_result = Verdict(animal="cat", confidence=0.4, at_bowl=True, reason="maybe")
    h.tick()
    assert h.events("veto") == []
    h.run(seconds=4)
    assert len(h.events("veto")) == 1


def test_cat_beside_grrr_is_logged_and_blocks_the_open(h: Harness) -> None:
    h.load_plates(1)
    h.llm.identify_result = Verdict(
        animal="grrr", confidence=0.95, at_bowl=True, other_animals_present=["cat"], reason="both"
    )
    h.run(seconds=12)
    assert [e.data["animal"] for e in h.events("veto")] == ["cat"]
    assert h.feeder.opens == []


def test_lingering_cat_writes_frames_only_for_logged_events(h: Harness) -> None:
    h.load_plates(1)
    h.llm.identify_result = CAT
    h.run(seconds=60)
    assert len(h.events("veto")) == 1
    assert len(list(h.machine.frames_dir.glob("*veto_cat*"))) == len(h.events("veto")[0].frames)


def test_done_still_logs_the_cat(h: Harness) -> None:
    h.config.update({"FEED_WITHOUT_LOADED_PLATES": False})  # classic: nothing marked = done
    h.tick(motion=False)
    assert h.state() == "DONE"
    h.llm.identify_result = CAT
    h.run(seconds=10)
    assert [e.data["animal"] for e in h.events("sighting")] == ["cat"]
    assert h.events("wanted_food_none_left") == []
    assert h.feeder.opens == []


def test_restart_mid_session_cannot_double_feed(h: Harness) -> None:
    h.load_plates(1, 2)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    h.machine = h.new_machine()
    h.machine.start()
    assert h.state() == "FEEDING"
    h.llm.feeding_result = EATING
    h.run(seconds=30)
    assert h.feeder.opens == ["open:1"]
    assert any("resumed FEEDING" in s for s in h.notifier.sent)


def test_restart_after_feed_sent_but_before_record_resumes_feeding(h: Harness) -> None:
    h.load_plates(1)
    h.store.set("state", "OPENING")
    h.machine = h.new_machine()
    h.machine.start()
    h.run(seconds=5)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == []


def test_feed_failure_notifies_and_returns_to_idle_without_retry(h: Harness) -> None:
    h.load_plates(1)
    h.feeder.fail_open = True
    confirm_grrr(h)
    assert h.state() in ("IDLE", "WATCHING")
    assert len(h.events("feed_failed")) == 1
    assert any("FEED FAILED" in s for s in h.notifier.sent)
    assert h.store.get_int("feeds_this_window") == 0


def test_done_reports_hungry_grrr_once_per_hour(h: Harness) -> None:
    h.config.update({"FEED_WITHOUT_LOADED_PLATES": False})  # classic: nothing marked = done
    h.tick(motion=False)
    assert h.state() == "DONE"
    h.llm.identify_result = GRRR
    h.run(seconds=10)
    assert len(h.events("wanted_food_none_left")) == 1
    assert h.llm.identify_calls == 1
    h.run(seconds=30, motion=False)
    h.run(seconds=10)
    assert h.llm.identify_calls == 2
    assert len(h.events("wanted_food_none_left")) == 1
    assert h.feeder.opens == []


def test_setting_plates_from_dashboard_leaves_done(h: Harness) -> None:
    h.config.update({"FEED_WITHOUT_LOADED_PLATES": False})  # classic: nothing marked = done
    h.tick(motion=False)
    assert h.state() == "DONE"
    h.store.set_plate(3, "loaded", h.clock.now)
    h.store.set("plates_updated", 1)
    h.tick(motion=False)
    assert h.state() == "IDLE"


def test_manual_feed_now_opens_whatever_plate_is_under_the_lid(h: Harness) -> None:
    h.tick(motion=False)
    h.machine.request_feed_now()
    h.tick(motion=False)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1"]


def test_dry_run_never_touches_the_real_feeder(h: Harness) -> None:
    h.config.update({"DRY_RUN": True})
    h.load_plates(1)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.calls == []
    assert h.events("open")[0].data["mode"] == "DRY_RUN"


def test_watching_returns_to_idle_without_motion(h: Harness) -> None:
    h.load_plates(1)
    h.tick(motion=True)
    assert h.state() == "WATCHING"
    h.run(seconds=25, motion=False)
    assert h.state() == "IDLE"


def test_camera_offline_is_reported_once(h: Harness) -> None:
    h.load_plates(1)
    h.tick(motion=False)
    h.clock.advance(seconds=120)
    h.machine.tick()
    h.clock.advance(seconds=10)
    h.machine.tick()
    assert len(h.events("camera_offline")) == 1
    h.tick(motion=False)
    assert len(h.events("camera_online")) == 1


def test_manual_rotate_runs_on_tick_and_logs(h: Harness) -> None:
    h.machine.request_manual_action("rotate")
    h.tick(motion=False)
    assert "rotate" in h.feeder.calls
    assert len(h.events("manual_rotate")) == 1


def test_manual_open_and_close_route_through_the_machine(h: Harness) -> None:
    h.machine.request_manual_action("open")
    h.tick(motion=False)
    assert h.feeder.opens == ["open:1"]
    assert len(h.events("manual_open")) == 1
    h.machine.request_manual_action("close")
    h.tick(motion=False)
    assert "close" in h.feeder.calls
    assert len(h.events("manual_close")) == 1


def test_manual_action_honors_dry_run(h: Harness) -> None:
    h.config.update({"DRY_RUN": True})
    h.machine.request_manual_action("open")
    h.tick(motion=False)
    assert h.feeder.calls == []  # real feeder untouched in dry run
    assert h.events("manual_open")[0].data["mode"] == "DRY_RUN"


def test_manual_action_failure_logs_and_does_not_raise(h: Harness) -> None:
    h.feeder.fail_open = True
    h.machine.request_manual_action("open")
    h.tick(motion=False)
    assert len(h.events("manual_open_failed")) == 1
    assert h.events("manual_open") == []


def test_unknown_manual_action_rejected(h: Harness) -> None:
    import pytest

    with pytest.raises(ValueError):
        h.machine.request_manual_action("explode")


def test_two_compartment_absence_algorithm(h: Harness) -> None:
    """The owner's POC algorithm: open 1 -> gone 10 min -> close -> return opens+rotates to
    2 -> gone 10 min -> close -> DONE, never opening again. LEAVE_TIMEOUT_S=600, MIN_GAP_MIN=0."""
    h.config.update({"LEAVE_TIMEOUT_S": 600, "MIN_GAP_MIN": 0})
    h.load_plates(1, 2)

    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1"]

    # Gone just over 10 minutes -> close plate 1.
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=610, motion=False)
    assert "close" in h.feeder.calls
    h.tick(motion=False)
    assert h.state() == "IDLE"  # MIN_GAP_MIN=0: ready again immediately, no hour-long wait

    # Dog returns -> tray rotates to compartment 2 and opens.
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1", "open:2"]
    assert "rotate" in h.feeder.calls

    # Gone 10 min again -> close plate 2 -> no plates left -> DONE.
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=610, motion=False)
    h.tick(motion=False)
    assert h.state() == "DONE"
    assert h.store.get_int("feeds_this_window") == 2

    # Terminated: even a confirmed dog does not open a third time.
    confirm_grrr(h)
    assert h.state() == "DONE"
    assert h.feeder.opens == ["open:1", "open:2"]


class _FakeEvents:
    def __init__(self):
        self.last_error = None
        self.motion_state = False
        self.animal_state = False
        self.on = False

    def motion_within(self, now, hold_s):
        return self.on


def test_camera_events_drive_motion_when_source_is_camera(h: Harness) -> None:
    ev = _FakeEvents()
    h.machine.events = ev
    h.config.update({"MOTION_SOURCE": "camera"})
    h.load_plates(1)
    h.tick(motion=False)  # frame-diff off, camera quiet
    assert h.state() == "IDLE"
    ev.on = True
    h.tick(motion=False)  # camera reports motion -> WATCHING (no frame-diff needed)
    assert h.state() == "WATCHING"


def test_falls_back_to_frame_diff_when_events_unhealthy(h: Harness) -> None:
    ev = _FakeEvents()
    ev.last_error = "subscription failed"
    ev.on = True  # would say motion, but it's broken so must be ignored
    h.machine.events = ev
    h.config.update({"MOTION_SOURCE": "camera"})
    h.load_plates(1)
    h.tick(motion=False)
    assert h.state() == "IDLE"  # broken events ignored, frame-diff quiet
    h.tick(motion=True)  # frame-diff fallback still feeds the dog
    assert h.state() == "WATCHING"


def test_feed_failure_backs_off_before_retrying(h: Harness) -> None:
    h.feeder.fail_open = True
    h.load_plates(1)
    confirm_grrr(h)
    assert h.events("feed_failed")
    assert h.state() in ("IDLE", "WATCHING")
    # Immediately re-confirming must NOT hammer the feeder during the backoff window.
    opens_before = len(h.feeder.opens)
    confirm_grrr(h)
    assert len(h.feeder.opens) == opens_before  # blocked by FEED_RETRY_BACKOFF_S


def test_capture_mode_saves_animal_frames_unlabelled(h: Harness) -> None:
    from tests.conftest import FakeLLM

    gate = FakeLLM()
    gate.identify_result = GRRR  # the local gate sees an animal
    h.machine.capture_gate = gate
    h.config.update({"CAPTURE_MODE": True})
    h.load_plates(1)
    h.tick(motion=True)
    saved = list((h.tmp_path / "unlabeled").glob("*.jpg"))
    assert len(saved) == 1
    assert h.store.get_int("capture_count") == 1
    # Capture only records frames; it must never move the feeder.
    assert h.feeder.opens == []


def test_capture_mode_skips_when_no_animal(h: Harness) -> None:
    from tests.conftest import UNSURE, FakeLLM

    gate = FakeLLM()
    gate.identify_result = UNSURE  # motion but nothing the detector recognises
    h.machine.capture_gate = gate
    h.config.update({"CAPTURE_MODE": True})
    h.load_plates(1)
    h.tick(motion=True)
    assert not (h.tmp_path / "unlabeled").exists() or not list(
        (h.tmp_path / "unlabeled").glob("*.jpg")
    )


def test_capture_mode_off_saves_nothing(h: Harness) -> None:
    from tests.conftest import FakeLLM

    gate = FakeLLM()
    gate.identify_result = GRRR
    h.machine.capture_gate = gate
    h.load_plates(1)
    h.tick(motion=True)
    assert not (h.tmp_path / "unlabeled").exists()


def test_capture_mode_throttles(h: Harness) -> None:
    from tests.conftest import FakeLLM

    gate = FakeLLM()
    gate.identify_result = CAT
    h.machine.capture_gate = gate
    h.config.update({"CAPTURE_MODE": True, "CAPTURE_MIN_GAP_S": 10})
    h.load_plates(1)
    h.tick(motion=True)  # saved
    h.tick(motion=True, seconds=2)  # 2s later, within the 10s gap -> skipped
    assert len(list((h.tmp_path / "unlabeled").glob("*.jpg"))) == 1
    h.tick(motion=True, seconds=10)  # now past the gap -> saved again
    assert len(list((h.tmp_path / "unlabeled").glob("*.jpg"))) == 2


def test_capture_mode_runs_outside_active_window(h: Harness) -> None:
    from tests.conftest import FakeLLM

    gate = FakeLLM()
    gate.identify_result = BOWIE
    h.machine.capture_gate = gate
    h.config.update({"CAPTURE_MODE": True})
    h.clock.now = datetime(2026, 9, 24, 13, 0)  # 1pm, well outside 21:00-06:00
    h.load_plates(1)
    h.tick(motion=True)
    assert h.state() == "OUTSIDE_WINDOW"  # feeding logic is dormant...
    assert len(list((h.tmp_path / "unlabeled").glob("*.jpg"))) == 1  # ...but capture still runs


def _open_with_verify(h: Harness, *plates: int) -> None:
    h.config.update({"VERIFY_FOOD": True})
    h.load_plates(*(plates or (1,)))
    confirm_grrr(h)  # motion + 3x Grrr -> OPENING -> VERIFYING (lid open, not yet feeding)


def test_verify_food_present_proceeds_to_feeding(h: Harness) -> None:
    h.llm.verify_result = HAS_FOOD
    _open_with_verify(h, 1)
    assert h.state() == "VERIFYING"
    assert h.feeder.opens == ["open:1"]
    h.run(seconds=15)  # first food check fires (>= VERIFY_POLL_S)
    assert h.state() == "FEEDING"
    assert "rotate" not in h.feeder.calls  # a full plate is never rotated away
    assert h.llm.verify_calls >= 1


def test_verify_empty_plate_rotates_to_a_full_one(h: Harness) -> None:

    # First plate reads empty twice, then the next plate has food.
    h.llm.verify_results = [NO_FOOD, NO_FOOD, HAS_FOOD]
    _open_with_verify(h, 1, 2)
    h.run(seconds=40)
    assert h.events("rotated_empty_plate"), "should have rotated off the empty plate"
    assert "open:2" in h.feeder.opens  # re-opened the next plate
    assert "rotate" in h.feeder.calls
    assert h.store.plates()[1] == "empty"  # the empty plate is recorded so it isn't re-offered
    assert h.state() == "FEEDING"


def test_verify_all_empty_alerts_and_closes(h: Harness) -> None:
    h.llm.verify_result = NO_FOOD  # every plate reads empty
    h.config.update({"MAX_ROTATE_FOR_FOOD": 2})
    _open_with_verify(h, 1, 2)
    h.run(seconds=80)
    assert h.events("empty_no_food"), "should alert when no full plate is found"
    assert h.state() in ("CLOSING", "COOLDOWN", "DONE")  # lid not left open on an empty plate


def test_verify_unsure_never_rotates_and_times_out_to_feeding(h: Harness) -> None:

    h.llm.verify_result = FOOD_UNSURE
    h.config.update({"VERIFY_TIMEOUT_S": 30})
    _open_with_verify(h, 1, 2)
    h.run(seconds=45)
    assert not h.events("rotated_empty_plate")  # unsure must never rotate the tray
    assert "rotate" not in h.feeder.calls
    assert h.state() == "FEEDING"  # don't starve Grrr: feed once verification gives up
    assert h.store.plates()[1] == "loaded"


def test_verify_off_opens_straight_to_feeding(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.llm.verify_calls == 0


def test_feed_failure_sends_an_urgent_alert(h: Harness) -> None:
    h.feeder.fail_open = True
    h.load_plates(1)
    confirm_grrr(h)
    assert any("FEED FAILED" in a for a in h.notifier.alerts)


def test_camera_offline_sends_an_urgent_alert(h: Harness) -> None:
    h.clock.advance(seconds=61)  # no frames pushed -> camera considered offline
    h.machine.tick()
    assert any("CAMERA OFFLINE" in a for a in h.notifier.alerts)


def test_heartbeat_pings_the_dead_mans_switch(h: Harness) -> None:
    pings: list[int] = []
    h.machine.healthcheck = lambda: pings.append(1)
    h.load_plates(1)
    h.tick()  # first tick fires the heartbeat
    assert pings, "heartbeat should ping the external healthcheck"


def watching_harness(tmp_path, start: datetime = datetime(2026, 9, 24, 12, 0)) -> Harness:
    """A harness with the free local detector wired in, at noon (off hours) by default."""
    h = Harness(tmp_path, start=start)
    h.capture_gate = FakeLLM()
    h.machine = h.new_machine()
    return h


def test_watch_logs_the_cat_off_hours_without_motion_or_paid_calls(tmp_path) -> None:
    h = watching_harness(tmp_path)
    h.load_plates(1)
    h.capture_gate.identify_result = CAT
    h.run(seconds=10, motion=False)  # a cat sitting still, staring
    assert h.state() == "OUTSIDE_WINDOW"
    assert [e.data["animal"] for e in h.events("sighting")] == ["cat"]
    assert h.events("sighting")[0].frames
    assert h.machine.last_verdict.animal == "cat"  # the dashboard sees her too
    assert h.llm.identify_calls == 0  # nothing paid off hours
    assert h.feeder.calls == []


def test_watch_logs_grrr_off_hours_but_never_opens(tmp_path) -> None:
    h = watching_harness(tmp_path)
    h.load_plates(1)
    h.capture_gate.identify_result = GRRR
    h.run(seconds=60)
    assert [e.data["animal"] for e in h.events("sighting")] == ["grrr"]
    assert h.state() == "OUTSIDE_WINDOW"
    assert h.feeder.calls == []


def test_watch_runs_in_cooldown_and_idle_but_not_while_watching(tmp_path) -> None:
    h = watching_harness(tmp_path, start=datetime(2026, 9, 24, 22, 0))
    h.load_plates(1)
    h.tick(motion=False)
    assert h.state() == "IDLE"
    h.capture_gate.identify_result = CAT
    h.tick(motion=False, seconds=2)
    assert [e.data["animal"] for e in h.events("sighting")] == ["cat"]
    calls = h.capture_gate.identify_calls
    h.run(seconds=10)  # motion -> WATCHING: the machine's own check takes over
    assert h.state() == "WATCHING"
    assert h.capture_gate.identify_calls == calls


def test_watch_respects_interval_and_switch(tmp_path) -> None:
    h = watching_harness(tmp_path)
    h.config.update({"WATCH_INTERVAL_S": 5})
    h.run(seconds=10, motion=False)
    assert h.capture_gate.identify_calls == 2
    h.config.update({"ALWAYS_WATCH": False})
    h.run(seconds=10, motion=False)
    assert h.capture_gate.identify_calls == 2


def test_manual_close_says_so_when_nothing_was_sent(h: Harness) -> None:
    """The feeder's cloud can report no open lid while the lid is physically open. A Close that
    sent nothing must not be logged as a close."""
    h.feeder.nothing_to_close = True
    h.machine.request_manual_action("close")
    h.tick(motion=False)
    assert h.events("manual_close_noop")
    assert not h.events("manual_close")
    assert "close" not in h.feeder.calls


def test_session_close_records_when_no_command_was_sent(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    h.feeder.nothing_to_close = True
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=h.machine.settings.LEAVE_TIMEOUT_S + 30)
    close = h.events("close")[0]
    assert close.data["close_sent"] is False
    assert "nothing sent" in close.reason


def test_eating_during_plate_verification_counts_as_presence(h: Harness) -> None:
    """Claude saying Grrr is at the bowl while the plate is being verified must hold the lid
    open: the leave timer used to run from the moment of opening regardless."""
    # The empty-bowl close would end this early; this test is about the leave timer.
    h.config.update({"VERIFY_TIMEOUT_S": 60, "LEAVE_TIMEOUT_S": 90, "CLOSE_WHEN_EMPTY": False})
    h.llm.verify_result = FeedingVerdict(grrr_at_bowl=True, bowl="unsure", reason="dog hides bowl")
    _open_with_verify(h, 1)
    h.run(seconds=70)  # verification gives up after 60s -> FEEDING
    assert h.state() == "FEEDING"
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=40)  # 110s after opening: past LEAVE_TIMEOUT from open, not from last seen
    assert h.state() == "FEEDING"


def test_feeds_from_the_bowl_under_the_lid_when_nothing_is_marked(h: Harness) -> None:
    """Forgot to tick a bowl: Grrr confirmed still opens whatever is under the lid."""
    assert h.store.loaded_plates() == []
    h.feeder.plate = 2
    confirm_grrr(h)
    assert h.feeder.opens == ["open:2"]
    assert h.machine.plates_tried() == [2]
    assert h.state() == "FEEDING"


def test_untried_bowls_are_tried_in_turn_until_all_are_empty(h: Harness) -> None:
    """Nothing marked and every bowl verified empty: rotate through all three once, then alert
    and finish for the window instead of churning the tray all night."""
    h.config.update({"VERIFY_FOOD": True, "MAX_ROTATE_FOR_FOOD": 2})
    h.llm.verify_result = NO_FOOD
    confirm_grrr(h)  # the tray starts on bowl 1
    h.run(seconds=90)
    assert h.feeder.opens == ["open:1", "open:2", "open:3"]
    assert sorted(h.machine.plates_tried()) == [1, 2, 3]
    assert h.events("empty_no_food")
    h.clock.advance(minutes=61)
    h.tick(motion=False)
    assert h.state() == "DONE"  # every bowl tried this window


def test_marked_bowls_are_honoured_exactly(h: Harness) -> None:
    """Marks from the dashboard win: once the marked bowl is eaten, no other bowl is tried."""
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=h.machine.settings.LEAVE_TIMEOUT_S + 30)
    h.clock.advance(minutes=61)
    h.tick(motion=False)
    assert h.state() == "DONE"
    assert h.feeder.opens == ["open:1"]


def test_a_new_window_forgets_which_bowls_were_tried(h: Harness) -> None:
    h.store.set("plates_tried", "[1, 2, 3]")
    h.store.set("state", "OUTSIDE_WINDOW")
    h.tick(motion=False)  # 22:00 is inside the window -> IDLE, with the slate wiped
    assert h.state() == "IDLE"
    assert h.machine.plates_tried() == []


NONE = Verdict(animal="none", confidence=0.95, at_bowl=False, reason="empty room")


def test_a_blank_frame_between_two_grrr_reads_does_not_reset_confirmation(h: Harness) -> None:
    """With the camera tight on the feeder she drops out of frame between reads. Two confident
    Grrr reads within CONFIRM_WINDOW_S open the lid even with 'none' in between."""
    h.load_plates(1)
    h.config.update({"CONFIRMATIONS_REQUIRED": 2})
    h.llm.identify_result = GRRR
    h.run(seconds=4)  # one read
    assert h.state() == "WATCHING"
    h.llm.identify_result = NONE
    h.run(seconds=6)  # two blank reads
    assert h.feeder.opens == []
    h.llm.identify_result = GRRR
    h.run(seconds=4)  # second confident read, 10 s after the first
    assert h.feeder.opens == ["open:1"]


def test_unsure_between_grrr_reads_does_not_reset_either(h: Harness) -> None:
    h.load_plates(1)
    h.config.update({"CONFIRMATIONS_REQUIRED": 2})
    h.llm.identify_result = GRRR
    h.run(seconds=4)
    h.llm.identify_result = UNSURE
    h.run(seconds=4)
    h.llm.identify_result = GRRR
    h.run(seconds=4)
    assert h.feeder.opens == ["open:1"]


def test_another_animal_between_grrr_reads_resets_confirmation(h: Harness) -> None:
    h.load_plates(1)
    h.config.update({"CONFIRMATIONS_REQUIRED": 2})
    h.llm.identify_result = GRRR
    h.run(seconds=4)
    h.llm.identify_result = BOWIE
    h.run(seconds=4)
    h.llm.identify_result = GRRR
    h.run(seconds=4)  # only one Grrr read since Bowie: not enough
    assert h.feeder.opens == []
    h.run(seconds=4)
    assert h.feeder.opens == ["open:1"]


def test_grrr_reads_older_than_the_window_do_not_count(h: Harness) -> None:
    h.load_plates(1)
    h.config.update({"CONFIRMATIONS_REQUIRED": 2, "CONFIRM_WINDOW_S": 30})
    h.llm.identify_result = GRRR
    h.run(seconds=4)
    h.llm.identify_result = NONE
    h.run(seconds=40)  # the first read ages out
    h.llm.identify_result = GRRR
    h.run(seconds=4)
    assert h.feeder.opens == []  # one fresh read only
    h.run(seconds=4)
    assert h.feeder.opens == ["open:1"]


def test_an_empty_session_skips_the_cooldown_so_the_next_bowl_can_be_tried(h: Harness) -> None:
    """Last night: bowl 3 was empty, the swap to the next bowl was refused by the device, and the
    machine sat in a 60-minute cooldown while she scratched at the lid. Nothing was served, so
    MIN_GAP_MIN must not apply; the device gets FEED_RETRY_BACKOFF_S, then the next bowl."""
    h.config.update({"VERIFY_FOOD": True, "CONFIRMATIONS_REQUIRED": 2})
    h.llm.verify_result = NO_FOOD
    h.feeder.fail_close = True  # the device refuses mid-swap, like motor error 2043
    confirm_grrr(h)
    assert h.feeder.opens == ["open:1"]  # nothing marked: the bowl under the lid
    h.run(seconds=25)  # two empty reads -> swap -> refused
    assert h.events("empty_no_food")
    assert h.state() in ("IDLE", "WATCHING"), "no food went out, so no cooldown"
    assert h.store.plates()[1] == "eaten"
    assert h.machine.next_allowed_feed() is None

    h.feeder.fail_close = False
    confirm_grrr(h)
    assert h.feeder.opens == ["open:1"], "backs off after the device refused"
    assert h.events("grrr_blocked")
    h.clock.advance(seconds=121)
    confirm_grrr(h)
    assert h.feeder.opens == ["open:1", "open:2"]  # the next untried bowl, this time with food?
    h.llm.verify_result = HAS_FOOD
    h.run(seconds=15)
    assert h.state() == "FEEDING"
    assert h.machine.next_allowed_feed() is not None  # now a real feed starts the gap


def test_a_fed_session_still_cools_down(h: Harness) -> None:
    h.load_plates(1, 2)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=125)
    assert h.state() == "COOLDOWN"
    assert h.events("transition")[0].data["fed"] is True


def reset_landed_in_idle(h: Harness) -> bool:
    """The reset itself goes to IDLE; motion still in the hold window may then start WATCHING."""
    transitions = h.store.events(kinds=("transition",))
    return any(
        e.reason == "cycle reset from dashboard" and e.data["to"] == "IDLE" for e in transitions
    ) and h.state() in ("IDLE", "WATCHING")


def test_reset_cycle_while_feeding_closes_and_starts_the_night_over(h: Harness) -> None:
    h.load_plates(1, 2)
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    h.machine.request_reset_cycle()
    h.tick(motion=False)
    assert reset_landed_in_idle(h)
    assert h.feeder.calls[-1] == "close"
    assert h.store.plates()[1] == "loaded"
    assert h.store.get_int("feeds_this_window") == 0
    assert h.store.get("last_fed_at") is None
    assert h.machine.plates_tried() == []
    event = h.events("cycle_reset")[0]
    # The lid was open, so plate 1 was still marked loaded; nothing needed restoring.
    assert event.data["closed"] is True and event.data["restored_plates"] == []
    assert any("CYCLE RESET" in s for s in h.notifier.sent)
    # Nothing blocks a fresh feed, and it comes from plate 1 again.
    confirm_grrr(h)
    assert h.state() == "FEEDING"
    assert h.feeder.opens == ["open:1", "open:1"]


def test_reset_cycle_in_cooldown_restores_the_eaten_bowl_and_clears_the_gap(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.feeding_result = GONE_EMPTY
    h.run(seconds=125)
    assert h.state() == "COOLDOWN"
    assert h.store.plates()[1] == "eaten"
    h.machine.request_reset_cycle()
    h.tick(motion=False)
    assert reset_landed_in_idle(h)
    assert h.store.loaded_plates() == [1]
    assert h.events("cycle_reset")[0].data["lid_was_open"] is False
    assert h.feeder.calls.count("close") == 1  # the lid was already closed; no second close
    confirm_grrr(h)
    assert h.state() == "FEEDING"


def test_reset_cycle_leaves_a_verified_empty_bowl_empty(h: Harness) -> None:
    h.load_plates(1)
    h.store.set("plates_tried", "[1]")
    h.store.set_plate(1, "empty", h.clock.now)
    h.tick(motion=False)
    h.machine.request_reset_cycle()
    h.tick(motion=False)
    assert h.store.plates()[1] == "empty"
    assert h.events("cycle_reset")[0].data["restored_plates"] == []


def test_reset_cycle_reports_a_failed_close(h: Harness) -> None:
    h.load_plates(1)
    confirm_grrr(h)
    h.feeder.fail_close = True
    h.machine.request_reset_cycle()
    h.tick(motion=False)
    assert reset_landed_in_idle(h)
    event = h.events("cycle_reset")[0]
    assert event.data["closed"] is False and "close failed" in event.data["close_error"]
    assert any("CLOSE FAILED" in s for s in h.notifier.sent)
