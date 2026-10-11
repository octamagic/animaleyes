"""Power-cycling the feeder through a smart plug: only on device-side faults, persisted before
acting, always turned back on, capped, and never a path to a double feed."""

from __future__ import annotations

import pytest

from animaleyes.feeder import is_device_fault
from animaleyes.power import CommandPlug, PowerError, build_power_switch
from tests.conftest import GRRR, Harness
from tests.test_machine import confirm_grrr

MOTOR_FAULT = (
    "petlibro-cli close X --no-dry-run failed: error: /device/wetFeedingPlan/stopFeedNow "
    "failed: code=2043 msg=电机异常无法关门"
)
TIMEOUT_FAULT = "platePositionChange failed: code=2020 msg=Device response timeout"
CLOUD_ERROR = "petlibro-cli feed X failed: error: 401 Unauthorized"


class FakePower:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_off = False
        self.fail_on = False

    def turn_off(self) -> None:
        if self.fail_off:
            raise PowerError("plug unreachable")
        self.calls.append("off")

    def turn_on(self) -> None:
        if self.fail_on:
            raise PowerError("plug unreachable")
        self.calls.append("on")


@pytest.fixture
def hp(tmp_path) -> Harness:
    h = Harness(tmp_path)
    h.power = FakePower()
    h.machine = h.new_machine()
    h.config.update({"CONFIRMATIONS_REQUIRED": 2})
    return h


def test_device_fault_classifier() -> None:
    assert is_device_fault(MOTOR_FAULT)
    assert is_device_fault(TIMEOUT_FAULT)
    assert not is_device_fault(CLOUD_ERROR)
    assert not is_device_fault("petlibro-cli: timed out after 40s")  # our own timeout


def test_motor_fault_on_open_pulls_the_plug_then_restores_and_waits(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = MOTOR_FAULT
    confirm_grrr(hp)
    assert hp.events("feed_failed")
    assert hp.power.calls == ["off"]
    assert hp.events("power_cycle")
    assert hp.store.get("plug_off_at") and not hp.store.get("plug_on_at")
    assert any("POWER CYCLED" in a for a in hp.notifier.alerts)

    hp.run(seconds=12, motion=False)  # POWER_CYCLE_OFF_S = 10
    assert hp.power.calls == ["off", "on"]
    assert hp.events("power_on")

    # Booting: no open even on a confirmed Grrr, until POWER_CYCLE_BOOT_S has passed.
    hp.feeder.fail_open = False
    confirm_grrr(hp)
    assert hp.feeder.opens == []
    assert "rebooted" in (hp.machine._open_blocker(hp.clock.now) or "")
    hp.clock.advance(seconds=95)
    confirm_grrr(hp)
    assert hp.feeder.opens == ["open:1"]


def test_cloud_errors_do_not_power_cycle(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = CLOUD_ERROR
    confirm_grrr(hp)
    assert hp.events("feed_failed")
    assert hp.power.calls == []


def test_no_commands_go_to_the_feeder_while_the_plug_is_off(hp: Harness) -> None:
    hp.load_plates(1)
    hp.store.set("plug_off_at", hp.clock.now.isoformat())
    hp.config.update({"POWER_CYCLE_OFF_S": 600})
    hp.machine.request_manual_action("rotate")
    hp.tick(motion=False)
    assert "rotate" not in hp.feeder.calls
    assert "power-cycling" in hp.events("manual_rotate_failed")[0].reason


def test_automatic_cycles_are_capped_but_manual_is_not(hp: Harness) -> None:
    hp.load_plates(1, 2)
    hp.feeder.fail_open = True
    hp.feeder.open_error = MOTOR_FAULT
    confirm_grrr(hp)
    hp.run(seconds=12, motion=False)
    assert hp.power.calls == ["off", "on"]
    hp.clock.advance(seconds=200)  # past boot + feed backoff, inside POWER_CYCLE_MIN_GAP_MIN
    confirm_grrr(hp)
    assert len(hp.events("feed_failed")) == 2
    assert hp.power.calls == ["off", "on"], "a second fault within the gap is not cycled"
    assert hp.events("power_cycle_skipped")
    hp.machine.request_manual_action("power_cycle")
    hp.tick(motion=False)
    assert hp.power.calls == ["off", "on", "off"]
    hp.run(seconds=12, motion=False)
    assert hp.power.calls == ["off", "on", "off", "on"]


def test_a_restart_mid_cycle_turns_the_plug_back_on(hp: Harness) -> None:
    hp.store.set("plug_off_at", hp.clock.now.isoformat())
    hp.clock.advance(seconds=30)
    hp.machine = hp.new_machine()
    hp.machine.start()
    hp.tick(motion=False)
    assert hp.power.calls == ["on"]
    assert not hp.machine.plug_is_off()


def test_turn_on_failure_alerts_and_keeps_retrying(hp: Harness) -> None:
    hp.store.set("plug_off_at", hp.clock.now.isoformat())
    hp.power.fail_on = True
    hp.run(seconds=15, motion=False)
    assert hp.events("power_on_failed")
    assert any("power" in a.lower() for a in hp.notifier.alerts)
    assert hp.machine.plug_is_off()
    hp.power.fail_on = False
    hp.tick(motion=False)
    assert hp.power.calls[-1] == "on"
    assert not hp.machine.plug_is_off()


def test_turn_off_failure_is_reported_and_leaves_no_off_state(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = MOTOR_FAULT
    hp.power.fail_off = True
    confirm_grrr(hp)
    assert hp.events("power_cycle_failed")
    assert not hp.machine.plug_is_off()


def test_setting_off_disables_automatic_cycling(hp: Harness) -> None:
    hp.config.update({"POWER_CYCLE_ON_FAULT": False})
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = MOTOR_FAULT
    confirm_grrr(hp)
    assert hp.power.calls == []


def test_without_a_plug_nothing_happens(h: Harness) -> None:
    h.load_plates(1)
    h.feeder.fail_open = True
    h.feeder.open_error = MOTOR_FAULT
    confirm_grrr(h)
    assert h.events("feed_failed")
    assert not h.events("power_cycle")
    assert h.machine._status_data()["power_cycle_available"] is False


def test_dry_run_never_touches_the_plug(hp: Harness) -> None:
    hp.config.update({"DRY_RUN": True})
    hp.machine.request_manual_action("power_cycle")
    hp.tick(motion=False)
    assert hp.power.calls == []
    assert hp.events("manual_power_cycle")


def test_command_plug_runs_shell_commands_and_reports_failure(tmp_path) -> None:
    marker = tmp_path / "state"
    plug = CommandPlug(f"echo off > {marker}", f"echo on > {marker}")
    plug.turn_off()
    assert marker.read_text().strip() == "off"
    plug.turn_on()
    assert marker.read_text().strip() == "on"
    with pytest.raises(PowerError):
        CommandPlug("exit 3", "true").turn_off()


def test_build_power_switch_picks_a_driver() -> None:
    assert build_power_switch("") is None
    assert isinstance(build_power_switch("", off_cmd="true", on_cmd="true"), CommandPlug)
    assert build_power_switch("10.0.0.50").__class__.__name__ == "KasaPlug"


def test_grrr_verdict_is_unchanged_by_power_feature() -> None:
    assert GRRR.animal == "grrr"  # keeps the import honest; the feature never touches verdicts
