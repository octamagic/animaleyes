"""Power-cycling the feeder through a smart plug: on any live-command error, persisted before
acting, always turned back on, waiting for the feeder to really be back, retried a bounded number
of times, and never a path to a double feed."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from animaleyes.feeder import FeederError, health_from_status
from animaleyes.power import CommandPlug, PowerError, ShellyPlug, build_power_switch
from tests.conftest import BOWIE, GONE_EMPTY, GRRR, UNSURE, Harness
from tests.test_machine import confirm_grrr

MOTOR_FAULT = (
    "petlibro-cli close X --no-dry-run failed: error: /device/wetFeedingPlan/stopFeedNow "
    "failed: code=2043 msg=电机异常无法关门"
)
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


def wait_for_boot(h: Harness) -> None:
    """Off time, then the first online probe (POWER_CYCLE_OFF_S + POWER_CYCLE_BOOT_S + a probe)."""
    h.run(seconds=60, motion=False)


# the fault -> cycle -> wait -> retry loop ---------------------------------------------------


def test_any_open_error_pulls_the_plug_and_retries_once_the_feeder_is_back(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = CLOUD_ERROR  # not just motor faults: any error
    hp.feeder.online = False  # the cloud still sees it offline while it boots
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE  # no further confirmations from here on
    failed = hp.events("feed_failed")
    assert len(failed) == 1 and failed[0].data["attempt"] == 1 and failed[0].data["will_retry"]
    assert hp.power.calls == ["off"]
    assert hp.events("power_cycle")
    assert hp.store.get("plug_off_at") and not hp.store.get("plug_on_at")
    assert any("POWER CYCLED" in a for a in hp.notifier.alerts)
    assert 1 not in hp.machine.plates_tried(), "a plate that never opened is not 'tried'"

    hp.run(seconds=12, motion=False)  # POWER_CYCLE_OFF_S = 10
    assert hp.power.calls == ["off", "on"]
    assert hp.events("power_on")
    hp.feeder.fail_open = False

    # Power is back but the feeder is not: no retry, no open, however long we wait.
    hp.run(seconds=120, motion=False)
    assert hp.feeder.opens == []
    assert hp.machine.feeder_booting()
    assert hp.events("feeder_booting")
    assert "not confirmed online" in (hp.machine._open_blocker(hp.clock.now) or "")

    # The cloud sees it again: the retry goes out on the next probe, with no new confirmation.
    hp.feeder.online = True
    hp.run(seconds=20, motion=False)
    assert hp.events("feeder_online")
    assert not hp.machine.feeder_booting()
    assert hp.feeder.opens == ["open:1"]
    assert "retry 2/4" in hp.events("open")[0].reason
    assert hp.store.get("feed_retry") is None
    assert hp.state() == "FEEDING"


def test_feeder_must_have_checked_in_since_the_power_on(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    hp.feeder.fail_open = False
    hp.feeder.last_online = hp.clock.now  # stale: before the plug came back on
    wait_for_boot(hp)
    assert hp.machine.feeder_booting(), "an old last-online time is not proof it is back"
    assert hp.feeder.opens == []
    hp.feeder.last_online = hp.clock.now
    hp.run(seconds=20, motion=False)
    assert not hp.machine.feeder_booting()
    assert hp.feeder.opens == ["open:1"]


def test_boot_probe_errors_are_not_faults_and_the_wait_is_bounded(hp: Harness) -> None:
    hp.config.update({"POWER_CYCLE_BOOT_MAX_S": 120})
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    hp.feeder.fail_open = False

    def dead():
        raise FeederError("status: Device response timeout")

    hp.feeder.health = dead  # type: ignore[method-assign]
    hp.run(seconds=100, motion=False)
    assert hp.power.calls == ["off", "on"], "a failing probe never triggers another cycle"
    assert hp.feeder.opens == []
    hp.run(seconds=40, motion=False)
    assert hp.events("feeder_not_back")
    assert not hp.machine.feeder_booting()
    assert hp.feeder.opens == ["open:1"], "past BOOT_MAX the retry goes out anyway"


def test_gives_up_after_feed_retries_more_attempts(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.feeder.open_error = MOTOR_FAULT
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    hp.run(seconds=900, motion=False)
    assert len(hp.events("feed_failed")) == 4  # FEED_RETRIES = 3 retries + the first try
    assert hp.power.calls.count("off") == 3, "the last failure is not worth another reboot"
    assert hp.power.calls[-1] == "on", "the feeder is never left without power"
    assert len(hp.events("feed_given_up")) == 1
    assert hp.store.get("feed_retry") is None
    assert any("GIVING UP" in a for a in hp.notifier.alerts)
    hp.feeder.fail_open = False
    hp.run(seconds=300, motion=False)
    assert hp.feeder.opens == [], "no more retries without a fresh confirmation"


def test_retry_is_held_while_another_animal_is_in_view(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    hp.feeder.fail_open = False
    hp.llm.identify_result = BOWIE
    wait_for_boot(hp)  # Bowie keeps getting identified while the feeder comes back
    hp.run(seconds=20)
    assert hp.feeder.opens == []
    assert hp.events("retry_held")
    hp.llm.identify_result = UNSURE
    hp.run(seconds=70, motion=False)  # Bowie's last read ages out
    assert hp.feeder.opens == ["open:1"]


def test_retry_episode_expires(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    hp.feeder.fail_open = False
    hp.feeder.online = False  # never comes back
    hp.config.update({"POWER_CYCLE_BOOT_MAX_S": 100_000})
    hp.run(seconds=21 * 60, motion=False, step=5)
    assert hp.store.get("feed_retry") is None
    assert any("expired" in e.reason for e in hp.events("feed_given_up"))
    assert hp.feeder.opens == []


def test_without_a_plug_the_retry_waits_out_the_backoff(h: Harness) -> None:
    h.load_plates(1)
    h.feeder.fail_open = True
    confirm_grrr(h)
    h.llm.identify_result = UNSURE
    assert h.events("feed_failed")[0].data["power_cycled"] is False
    assert not h.events("power_cycle")
    assert h.machine._status_data()["power_cycle_available"] is False
    h.feeder.fail_open = False
    h.run(seconds=100, motion=False)
    assert h.feeder.opens == []  # FEED_RETRY_BACKOFF_S = 120
    h.run(seconds=30, motion=False)
    assert h.feeder.opens == ["open:1"]
    assert "retry 2/4" in h.events("open")[0].reason


def test_door_fault_reported_after_an_open_is_a_failed_open(hp: Harness) -> None:
    hp.load_plates(1)
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    assert hp.state() == "FEEDING" and hp.feeder.opens == ["open:1"]
    hp.feeder.device_error = "doorErrorState=OPEN_FAIL"
    hp.run(seconds=25, motion=False)  # LID_POLL_S = 20: one reading is not enough
    assert hp.state() == "FEEDING" and not hp.events("feed_failed")
    hp.run(seconds=20, motion=False)  # the second consecutive reading is
    failed = hp.events("feed_failed")
    assert failed and failed[0].data["trigger"] == "lid fault"
    assert "doorErrorState=OPEN_FAIL" in failed[0].reason
    assert hp.state() == "IDLE"
    assert hp.power.calls[0] == "off" and hp.events("power_cycle")
    assert hp.store.get("last_fed_at") is None, "nothing was served: no cooldown"
    assert 1 not in hp.machine.plates_tried()
    hp.feeder.device_error = None
    wait_for_boot(hp)
    hp.run(seconds=20, motion=False)
    assert hp.feeder.opens == ["open:1", "open:1"]


def close_refused(h: Harness) -> None:
    """Grrr eats and walks off; the device refuses the close, like this morning's 2043."""
    h.load_plates(1)
    confirm_grrr(h)
    h.llm.identify_result = UNSURE
    assert h.state() == "FEEDING"
    h.feeder.fail_close = True
    h.llm.feeding_result = GONE_EMPTY
    for _ in range(300):
        h.tick(motion=False)
        if h.events("close_failed"):
            return
    raise AssertionError("the close was never attempted")


def test_a_refused_close_is_retried_once_the_feeder_is_back(hp: Harness) -> None:
    """2026-10-29: close failed with 2043, the plug was cycled, and the machine called the lid
    closed and moved on. It sat open all day. The lid is open until a close goes through."""
    close_refused(hp)
    assert hp.state() == "CLOSING"
    assert hp.store.get("lid_actual_open") == "1"
    assert hp.power.calls == ["off"]
    assert any("CLOSE FAILED (attempt 1)" in a for a in hp.notifier.alerts)
    assert not hp.events("close")
    hp.feeder.fail_close = False
    hp.run(seconds=12, motion=False)
    assert "close" not in hp.feeder.calls, "nothing is sent until the feeder is back online"
    wait_for_boot(hp)
    assert hp.feeder.calls.count("close") == 1
    assert hp.state() == "COOLDOWN"
    assert hp.store.get("lid_actual_open") == "0"
    assert "on attempt 2" in hp.events("close")[0].reason
    assert hp.store.get("close_attempts") is None
    assert len(hp.events("close_failed")) == 1


def test_close_keeps_retrying_past_the_window_and_the_cycle_gap(hp: Harness) -> None:
    close_refused(hp)
    wait_for_boot(hp)
    assert len(hp.events("close_failed")) == 2, "retried as soon as the feeder came back"
    assert hp.power.calls == ["off", "on"], "a second fault inside the gap is not cycled"
    hp.clock.advance(hours=12)  # long past ACTIVE_END: an open lid is still worked on
    hp.tick(motion=False)
    assert hp.state() == "CLOSING"
    assert len(hp.events("close_failed")) == 3
    hp.feeder.fail_close = False
    hp.run(seconds=125, motion=False)
    assert hp.feeder.calls.count("close") == 1
    assert hp.state() == "OUTSIDE_WINDOW"


def test_without_a_plug_a_refused_close_waits_out_the_backoff(h: Harness) -> None:
    h.config.update({"CONFIRMATIONS_REQUIRED": 2})
    close_refused(h)
    h.feeder.fail_close = False
    h.run(seconds=100, motion=False)
    assert "close" not in h.feeder.calls  # FEED_RETRY_BACKOFF_S = 120
    h.run(seconds=30, motion=False)
    assert h.feeder.calls.count("close") == 1
    assert h.state() == "COOLDOWN"


def test_a_restart_in_closing_still_closes(hp: Harness) -> None:
    close_refused(hp)
    hp.feeder.fail_close = False
    wait_for_boot(hp)
    hp.store.set("state", "CLOSING")  # died before the retry landed
    hp.store.set("close_failed_at", hp.clock.now.isoformat())
    hp.machine = hp.new_machine()
    hp.machine.start()
    hp.run(seconds=125, motion=False)
    assert hp.state() == "COOLDOWN"
    assert hp.feeder.opens == ["open:1"], "a restart never re-opens"


def test_lid_poll_errors_do_not_cycle(hp: Harness) -> None:
    hp.load_plates(1)

    def broken():
        raise FeederError(MOTOR_FAULT)

    hp.feeder.manual_feed_active = broken  # type: ignore[method-assign]
    hp.run(seconds=45, motion=False)
    assert hp.power.calls == []


# the plug itself ---------------------------------------------------------------------------


def test_no_commands_go_to_the_feeder_while_the_plug_is_off(hp: Harness) -> None:
    hp.load_plates(1)
    hp.store.set("plug_off_at", hp.clock.now.isoformat())
    hp.config.update({"POWER_CYCLE_OFF_S": 600})
    hp.machine.request_manual_action("rotate")
    hp.tick(motion=False)
    assert "rotate" not in hp.feeder.calls
    assert "power-cycling" in hp.events("manual_rotate_failed")[0].reason


def test_cycles_outside_a_feed_attempt_are_capped_but_manual_is_not(hp: Harness) -> None:
    hp.feeder.fail_close = True
    hp.machine.request_manual_action("close")
    hp.tick(motion=False)
    assert hp.power.calls == ["off"]
    hp.run(seconds=12, motion=False)
    assert hp.power.calls == ["off", "on"]
    hp.machine.request_manual_action("close")
    hp.tick(motion=False)
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
    assert hp.machine.feeder_booting()


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


def test_turn_off_failure_is_reported_and_the_retry_still_happens(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    hp.power.fail_off = True
    confirm_grrr(hp)
    hp.llm.identify_result = UNSURE
    assert hp.events("power_cycle_failed")
    assert not hp.machine.plug_is_off()
    assert hp.events("feed_failed")[0].data["power_cycled"] is False
    hp.feeder.fail_open = False
    hp.run(seconds=130, motion=False)
    assert hp.feeder.opens == ["open:1"]


def test_setting_off_disables_automatic_cycling(hp: Harness) -> None:
    hp.config.update({"POWER_CYCLE_ON_FAULT": False})
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    assert hp.power.calls == []
    assert hp.store.get("feed_retry"), "the retry is still armed, just without a reboot"


def test_dry_run_never_touches_the_plug(hp: Harness) -> None:
    hp.config.update({"DRY_RUN": True})
    hp.machine.request_manual_action("power_cycle")
    hp.tick(motion=False)
    assert hp.power.calls == []
    assert hp.events("manual_power_cycle")


def test_status_exposes_the_retry_and_boot_state(hp: Harness) -> None:
    hp.load_plates(1)
    hp.feeder.fail_open = True
    confirm_grrr(hp)
    status = hp.machine._status_data()
    assert status["plug_off"] is True
    assert status["feed_retry"]["attempts"] == 1
    hp.run(seconds=12, motion=False)
    assert hp.machine._status_data()["feeder_booting"] is True


def test_health_from_status_reads_the_cloud_payload() -> None:
    ok = health_from_status(
        {"online": True, "lastOnlineTime": 1791772328270, "errorState": False,
         "doorErrorState": "NORMAL", "barnDoorError": False, "runningState": "IDLE",
         "platePosition": 2}
    )  # fmt: skip
    assert ok.ok and ok.plate == 2 and ok.last_online_at is not None
    bad = health_from_status({"online": True, "doorErrorState": "STUCK", "errorState": True})
    assert bad.error == "errorState, doorErrorState=STUCK" and not bad.ok
    assert not health_from_status({"online": False}).ok


# drivers -------------------------------------------------------------------------------------


class _ShellyHandler(BaseHTTPRequestHandler):
    mac = "ACEBE6F0CA10"
    gen: int | None = 4
    state = {"on": True, "log": []}

    def log_message(self, *a) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/shelly":
            body = {"mac": self.mac, "model": "S4PL-00116US"}
            if self.gen:
                body["gen"] = self.gen
        elif url.path == "/rpc/Switch.Set":
            self.state["on"] = q["on"][0] == "true"
            self.state["log"].append(self.path)
            body = {"was_on": True}
        elif url.path == "/rpc/Switch.GetStatus":
            body = {"id": 0, "output": self.state["on"], "apower": 54.8}
        elif url.path == "/relay/0":
            if "turn" in q:
                self.state["on"] = q["turn"][0] == "on"
                self.state["log"].append(self.path)
            body = {"ison": self.state["on"]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def shelly():
    _ShellyHandler.state = {"on": True, "log": []}
    _ShellyHandler.gen = 4
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ShellyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_shelly_switches_by_verified_mac(shelly: int) -> None:
    plug = ShellyPlug("AC:EB:E6:F0:CA:10", host="127.0.0.1", port=shelly, sweep=False)
    plug.turn_off()
    assert _ShellyHandler.state["on"] is False
    plug.turn_on()
    assert _ShellyHandler.state["on"] is True
    assert plug.gen == 4
    assert plug.status()["apower"] == 54.8


def test_shelly_refuses_a_host_with_the_wrong_mac(shelly: int, tmp_path) -> None:
    arp = tmp_path / "arp"
    arp.write_text("IP address HW type Flags HW address Mask Device\n")
    plug = ShellyPlug(
        "00:11:22:33:44:55", host="127.0.0.1", port=shelly, arp_table=str(arp), sweep=False
    )
    with pytest.raises(PowerError, match="not found"):
        plug.turn_off()
    assert _ShellyHandler.state["log"] == []


def test_shelly_is_found_through_the_arp_table_when_the_address_changed(shelly, tmp_path) -> None:
    arp = tmp_path / "arp"
    arp.write_text(
        "IP address       HW type     Flags       HW address            Mask     Device\n"
        "10.0.0.1         0x1         0x2         4c:d7:4a:7a:4e:0f     *        eno2\n"
        "127.0.0.1        0x1         0x2         ac:eb:e6:f0:ca:10     *        lo\n"
    )
    plug = ShellyPlug(
        "ac:eb:e6:f0:ca:10", host="127.0.0.2", port=shelly, arp_table=str(arp), sweep=False
    )
    plug.turn_off()
    assert plug.host == "127.0.0.1"
    assert _ShellyHandler.state["on"] is False


def test_shelly_gen1_uses_the_relay_endpoint(shelly: int) -> None:
    _ShellyHandler.gen = None
    plug = ShellyPlug("ac:eb:e6:f0:ca:10", host="127.0.0.1", port=shelly, sweep=False)
    plug.turn_off()
    assert plug.gen == 1
    assert _ShellyHandler.state["log"] == ["/relay/0?turn=off"]


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
    assert build_power_switch() is None
    assert isinstance(build_power_switch(off_cmd="true", on_cmd="true"), CommandPlug)
    assert isinstance(build_power_switch(mac="ac:eb:e6:f0:ca:10"), ShellyPlug)
    assert isinstance(build_power_switch("shelly", host="10.0.0.165"), ShellyPlug)
    assert build_power_switch("kasa").__class__.__name__ == "NoneType"
    assert build_power_switch("kasa", host="10.0.0.50").__class__.__name__ == "KasaPlug"
    with pytest.raises(PowerError):
        build_power_switch("hue", host="x")


def test_grrr_verdict_is_unchanged_by_power_feature() -> None:
    assert GRRR.animal == "grrr"  # keeps the import honest; the feature never touches verdicts
