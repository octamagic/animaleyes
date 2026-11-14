from __future__ import annotations

from datetime import time

import pytest
from PIL import Image

from animaleyes.camera import (
    arp_lookup,
    build_ffmpeg_command,
    kasa_legacy_password,
    resolve_stream_url,
    split_jpegs,
)
from animaleyes.config import Secrets, Settings
from animaleyes.motion import changed_fraction
from animaleyes.vision import Verdict, estimate_cost_usd


def test_active_window_crossing_midnight() -> None:
    s = Settings(ACTIVE_START="21:00", ACTIVE_END="06:00")
    assert s.is_active_at(time(23, 0))
    assert s.is_active_at(time(2, 0))
    assert not s.is_active_at(time(12, 0))
    assert not s.is_active_at(time(6, 0))


def test_active_window_same_day() -> None:
    s = Settings(ACTIVE_START="08:00", ACTIVE_END="17:00")
    assert s.is_active_at(time(9, 0))
    assert not s.is_active_at(time(20, 0))


def test_split_jpegs_handles_partial_tail() -> None:
    a = b"\xff\xd8AAA\xff\xd9"
    b = b"\xff\xd8BBB\xff\xd9"
    buf = bytearray(b"junk" + a + b + b"\xff\xd8partial")
    assert split_jpegs(buf) == [a, b]
    assert bytes(buf) == b"\xff\xd8partial"
    buf += b"\xff\xd9"
    assert split_jpegs(buf) == [b"\xff\xd8partial\xff\xd9"]


def secrets(url: str, email: str = "", password: str = "") -> Secrets:
    return Secrets(
        kasa_stream_url=url,
        kasa_email=email,
        kasa_password=password,
        anthropic_api_key="",
        petlibro_serial="",
        slack_webhook="",
        dash_user="",
        dash_password="",
        dash_public_url="",
    )


def test_ffmpeg_command_rtsp_uses_tcp() -> None:
    cmd = build_ffmpeg_command(secrets("rtsp://u:p@cam/stream1"))
    assert "-rtsp_transport" in cmd and "-headers" not in cmd
    assert cmd[-1] == "-"


def test_ffmpeg_command_legacy_kasa_uses_basic_auth_with_hashed_password() -> None:
    cmd = build_ffmpeg_command(secrets("https://cam:19443/https/stream/mixed", "me@x.com", "pw"))
    header = cmd[cmd.index("-headers") + 1]
    assert header.startswith("Authorization: Basic ")

    assert kasa_legacy_password("secret1") == "c2VjcmV0MQ=="  # go2rtc's documented example


def test_changed_fraction_zero_for_identical_and_one_for_inverted() -> None:
    black = Image.new("L", (64, 36), 0)
    white = Image.new("L", (64, 36), 255)
    assert changed_fraction(black, black) == 0.0
    assert changed_fraction(black, white) == 1.0


def test_verdict_is_grrr_requires_confidence_bowl_and_solitude() -> None:
    v = Verdict(animal="grrr", confidence=0.9, at_bowl=True)
    assert v.is_grrr(0.8)
    assert not Verdict(animal="grrr", confidence=0.7, at_bowl=True).is_grrr(0.8)
    assert not Verdict(animal="grrr", confidence=0.9, at_bowl=False).is_grrr(0.8)
    assert not Verdict(
        animal="grrr", confidence=0.9, at_bowl=True, other_animals_present=["cat"]
    ).is_grrr(0.8)


def test_cost_estimate_counts_cache_reads_cheaply() -> None:
    cold = estimate_cost_usd(
        "claude-opus-5",
        {"input_tokens": 1000, "cache_creation_input_tokens": 3000, "output_tokens": 100},
    )
    warm = estimate_cost_usd(
        "claude-opus-5",
        {"input_tokens": 1000, "cache_read_input_tokens": 3000, "output_tokens": 100},
    )
    assert warm < cold
    assert round(warm, 6) == round((1000 * 5 + 300 * 5 + 100 * 25) / 1e6, 6)


ARP = """IP address       HW type     Flags       HW address            Mask     Device
10.0.0.1         0x1         0x2         4c:d7:4a:7a:4e:0f     *        eno2
10.0.0.99        0x1         0x0         a8:42:a1:d1:d0:26     *        eno2
10.0.0.216       0x1         0x2         a8:42:a1:d1:d0:26     *        eno2
"""
CAM_MAC = "A8:42:A1:D1:D0:26"


def test_arp_lookup_skips_incomplete_entries_and_ignores_mac_case(tmp_path) -> None:
    table = tmp_path / "arp"
    table.write_text(ARP)
    assert arp_lookup(CAM_MAC, table) == "10.0.0.216"
    assert arp_lookup("00:00:00:00:00:00", table) is None


def test_stream_url_without_placeholder_is_used_as_is() -> None:
    url = "rtsp://u:p@10.0.0.5:554/stream1"
    assert resolve_stream_url(url, "", lookup=lambda m: 1 / 0) == url


def test_camera_is_found_by_mac_after_a_rescan_on_the_stream_port() -> None:
    table: dict[str, str] = {}
    knocked: list[int] = []

    def rescan(port: int) -> None:
        knocked.append(port)
        table[CAM_MAC] = "10.0.0.240"  # the camera took a new lease

    url = resolve_stream_url(
        "https://{host}:19443/https/stream/mixed", CAM_MAC, lookup=table.get, rescan=rescan
    )
    assert url == "https://10.0.0.240:19443/https/stream/mixed"
    assert knocked == [19443]


def test_missing_camera_or_mac_is_an_error_not_a_guess() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        resolve_stream_url("rtsp://{host}/s", CAM_MAC, lookup=lambda m: None, rescan=lambda p: None)
    with pytest.raises(RuntimeError, match="KASA_CAMERA_MAC"):
        resolve_stream_url("rtsp://{host}/s", "")


def test_feeder_auto_logs_in_when_no_token(monkeypatch) -> None:
    """A missing/expired token self-heals: the wrapper runs `login` and retries once."""
    import subprocess
    from types import SimpleNamespace

    from animaleyes.feeder import PetlibroCli

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        sub = cmd[1]
        if sub == "status" and calls.count(["petlibro-cli", "status", "AF0"]) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="error: No cached token. Run `petlibro-cli login` first.",
            )
        if sub == "login":
            return SimpleNamespace(returncode=0, stdout="Logged in", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"platePosition": 2}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert PetlibroCli("AF0").current_plate() == 2
    subs = [c[1] for c in calls]
    assert subs == ["status", "login", "status"]  # failed, logged in, retried


def test_feeder_raises_on_non_auth_error(monkeypatch) -> None:
    import subprocess
    from types import SimpleNamespace

    from animaleyes.feeder import FeederError, PetlibroCli

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="code=1005 msg=No resource access")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        PetlibroCli("AF0").current_plate()
        raise AssertionError("expected FeederError")
    except FeederError as exc:
        assert "No resource access" in str(exc)


def test_camera_credentials_parsed_from_rtsp_url() -> None:
    from animaleyes.events import camera_credentials

    # URL-encoded special chars must round-trip; value is a dummy, not a real credential.
    assert camera_credentials("rtsp://admin:p%40ss%21word@10.0.0.8:554/x") == (
        "admin",
        "p@ss!word",
    )


def test_events_apply_tracks_motion_and_animal() -> None:
    from datetime import datetime

    from animaleyes.events import ReolinkEvents

    e = ReolinkEvents("rtsp://admin:pw@10.0.0.8:554/x", "aa:bb:cc:dd:ee:ff")
    e._apply(
        "<wsnt:NotificationMessage><wsnt:Topic>tns1:RuleEngine/CellMotionDetector/Motion"
        '</wsnt:Topic><SimpleItem Name="IsMotion" Value="true"/></wsnt:NotificationMessage>'
    )
    assert e.motion_state is True
    assert e.motion_within(datetime.now(), 60) is True

    e._apply(
        "<wsnt:NotificationMessage><wsnt:Topic>tns1:RuleEngine/MyRuleDetector/DogCatDetect"
        '</wsnt:Topic><SimpleItem Name="State" Value="true"/></wsnt:NotificationMessage>'
    )
    assert e.animal_state is True


def test_open_now_treats_timeout_as_success_when_feed_active(monkeypatch) -> None:
    """A slow open that times out on the HTTP read is treated as success when the device
    confirms a manual feed is active — so the machine won't retry and double-feed."""
    import subprocess
    from types import SimpleNamespace

    from animaleyes.feeder import PetlibroCli

    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd[1])
        if cmd[1] == "feed":
            return SimpleNamespace(
                returncode=1, stdout="", stderr="HTTP error: The read operation timed out"
            )
        if cmd[1] == "close":  # manual_feed_active() dry-run plan check
            return SimpleNamespace(returncode=0, stdout='Response data: {"feedId": 123}', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    PetlibroCli("AF0").open_now(1)  # must not raise
    assert "feed" in seen and "close" in seen


def test_open_now_raises_on_timeout_when_no_feed_active(monkeypatch) -> None:
    import subprocess
    from types import SimpleNamespace

    from animaleyes.feeder import FeederError, PetlibroCli

    def fake_run(cmd, **kw):
        if cmd[1] == "feed":
            return SimpleNamespace(
                returncode=1, stdout="", stderr="HTTP error: The read operation timed out"
            )
        if cmd[1] == "close":
            return SimpleNamespace(
                returncode=1, stdout="", stderr="No active manual feed (manualFeedId is empty)."
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        PetlibroCli("AF0").open_now(1)
        raise AssertionError("expected FeederError")
    except FeederError:
        pass


def test_claude_rebuilds_reference_blocks_when_personas_change(tmp_path, monkeypatch) -> None:
    """Editing a persona from the dashboard takes effect: the cached prefix and schema are
    rebuilt with the new label/roster on the next identify."""
    from animaleyes.personas import Persona
    from animaleyes.vision import ClaudeIdentifier

    roster = [
        Persona("grrr", "Grrr", "v1", feedable=True),
        Persona("bowie", "Bowie", "b"),
        Persona("cat", "Chicken", "c"),
    ]
    for a in ("grrr", "bowie", "cat"):
        (tmp_path / a).mkdir()

    class _NoClient:
        pass

    ident = ClaudeIdentifier(
        tmp_path, store=None, model=lambda: "m", client=_NoClient(), personas=lambda: roster
    )
    assert any("v1" in b.get("text", "") for b in ident.reference_blocks)
    assert ident.target_name == "Grrr"
    assert "cat" in ident.identify_schema["properties"]["animal"]["enum"]
    # Rename the target and add a new animal; the next guard rebuilds everything.
    roster[0] = Persona("grrr", "Gizmo", "a new description", feedable=True)
    roster.append(Persona("rex", "Rex", "a big dog"))
    ident._maybe_reload()
    assert any("a new description" in b.get("text", "") for b in ident.reference_blocks)
    assert ident.target_name == "Gizmo"
    assert "rex" in ident.identify_schema["properties"]["animal"]["enum"]


def test_persona_store_crud_and_feed_invariant(tmp_path) -> None:
    from animaleyes.personas import PersonaStore

    store = PersonaStore(tmp_path / "personas.json")
    roster = store.load()  # seeds defaults
    names = {p.key: p.name for p in roster}
    assert names["cat"] == "Chicken"  # the cat's display name
    assert [p.key for p in roster if p.feedable] == ["grrr"]  # exactly the target is feedable

    # Add a new animal — always blocked, with a slug key.
    rex = store.upsert("Rex", "a big dog")
    assert rex.key == "rex" and rex.feedable is False
    assert "rex" in store.keys()

    # Rename the cat; key stays stable so photos/history don't move.
    store.upsert("Mr Whiskers", "sleek", key="cat")
    assert store.names()["cat"] == "Mr Whiskers"

    # Even if a tampered file marks another animal feedable, load() re-locks to the target only.
    (tmp_path / "personas.json").write_text(
        '[{"key":"grrr","name":"Grrr","feedable":false},'
        '{"key":"bowie","name":"Bowie","feedable":true}]'
    )
    reloaded = {p.key: p.feedable for p in store.load()}
    assert reloaded == {"grrr": True, "bowie": False}

    # The feedable target can never be deleted.
    import pytest

    with pytest.raises(ValueError):
        store.delete("grrr")


def test_theme_store_crud_and_validation(tmp_path) -> None:
    from animaleyes.themes import ThemeStore

    store = ThemeStore(tmp_path / "themes.json")
    assert store.load() == []  # no custom themes to start

    # Create: unknown tokens and bad hex values are dropped; a short id is assigned.
    t = store.upsert("Midnight", {"bg": "#0d0a1a", "teal": "nothex", "bogus": "#ffffff"})
    assert t.id and t.name == "Midnight"
    assert t.colors == {"bg": "#0d0a1a"}  # teal (bad hex) and bogus (unknown) dropped

    # Edit by id.
    store.upsert("Midnight 2", {"bg": "#010203", "teal": "#112233"}, theme_id=t.id)
    reloaded = store.load()
    assert len(reloaded) == 1
    assert reloaded[0].name == "Midnight 2"
    assert reloaded[0].colors == {"bg": "#010203", "teal": "#112233"}

    # Delete.
    store.delete(t.id)
    assert store.load() == []


def test_camera_pump_reconnects_on_a_stalled_stream(tmp_path) -> None:
    """A camera whose socket stays open but stops sending frames must not hang forever: the
    pump raises after stall_s so run()'s reconnect kicks in."""
    import os
    import time as _t

    from animaleyes.camera import Camera, FrameBuffer
    from animaleyes.config import Secrets

    r, w = os.pipe()
    rf = os.fdopen(r, "rb", buffering=0)
    cam = Camera(Secrets.from_env(), FrameBuffer(), stall_s=0.3)
    start = _t.monotonic()
    try:
        with pytest.raises(RuntimeError, match="no camera frames"):
            cam._pump(rf)
    finally:
        rf.close()
        os.close(w)
    assert _t.monotonic() - start < 3  # bailed promptly, didn't hang


def test_camera_pump_pushes_frames_then_reports_eof() -> None:
    """Normal path: JPEGs on the pipe become frames; a closed stream returns (EOF -> reconnect)."""
    import os

    from animaleyes.camera import Camera, FrameBuffer
    from animaleyes.config import Secrets

    tiny = bytes.fromhex("ffd8") + b"x" + bytes.fromhex("ffd9")
    r, w = os.pipe()
    rf = os.fdopen(r, "rb", buffering=0)
    os.write(w, tiny + tiny)
    os.close(w)  # EOF after the two frames
    buf = FrameBuffer()
    cam = Camera(Secrets.from_env(), buf, stall_s=5)
    try:
        cam._pump(rf)  # returns on EOF (no raise inside _pump)
    finally:
        rf.close()
    assert len(buf.latest(5)) == 2


def test_email_notifier_enabled_flag_and_routine_is_silent() -> None:
    from animaleyes.notify import EmailNotifier

    off = EmailNotifier("", 587, "", "", "", [])
    assert off.enabled is False
    off.alert("x")  # disabled: no SMTP, no raise
    off.send("x")

    on = EmailNotifier("smtp.example.com", 587, "u", "p", "u@x.com", ["a@b.com"])
    assert on.enabled is True
    on.send("routine")  # routine never emails (no SMTP attempted), must not raise


def test_multinotifier_fans_out_and_survives_a_bad_channel() -> None:
    from animaleyes.notify import LogNotifier, MultiNotifier

    class Boom:
        def send(self, t):
            raise RuntimeError("down")

        def alert(self, t):
            raise RuntimeError("down")

    sink = LogNotifier()
    m = MultiNotifier([Boom(), sink])
    m.send("hi")
    m.alert("oops")
    assert "hi" in sink.sent
    assert "oops" in sink.alerts


def test_pushover_payload_is_emergency_priority() -> None:
    from animaleyes.notify import PushoverNotifier

    off = PushoverNotifier("", "")
    assert off.enabled is False
    off.alert("x")  # disabled: no HTTP, no raise

    n = PushoverNotifier("tok", "usr", retry_s=10, expire_s=99999)
    assert n.enabled is True
    p = n._payload("camera offline")
    assert p["priority"] == 2  # emergency: repeats until acknowledged
    assert p["retry"] >= 30 and p["expire"] <= 10800  # clamped to Pushover's limits
    assert p["message"] == "camera offline"
    n.send("routine")  # routine never pushes, must not raise


def test_ffmpeg_command_for_a_second_camera_uses_its_own_url_and_mac() -> None:
    feeder = secrets("rtsp://u:p@feeder/stream1")
    cmd = build_ffmpeg_command(feeder, stream_url="rtsp://u:p@bed/sub")
    assert "rtsp://u:p@bed/sub" in cmd and "rtsp://u:p@feeder/stream1" not in cmd
    with pytest.raises(RuntimeError, match="KASA_CAMERA_MAC"):
        # {host} needs this camera's own MAC; the feeder's is never borrowed.
        build_ffmpeg_command(feeder, stream_url="rtsp://u:p@{host}/sub", mac="")
