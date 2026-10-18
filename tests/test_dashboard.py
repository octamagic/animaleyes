from __future__ import annotations

import asyncio
import secrets as secrets_lib
from datetime import datetime

from fastapi.testclient import TestClient

from animaleyes.camera import Frame, FrameBuffer
from animaleyes.config import Secrets
from animaleyes.dashboard import create_app, mjpeg
from tests.conftest import Harness

# Generated per run so no credential-looking literal lives in the repo.
PASSWORD = secrets_lib.token_urlsafe(16)


def client(h: Harness) -> TestClient:
    secrets = Secrets(
        kasa_stream_url="",
        kasa_email="",
        kasa_password="",
        anthropic_api_key="",
        petlibro_serial="",
        slack_webhook="",
        dash_user="me",
        dash_password=PASSWORD,
        dash_public_url="http://dash",
    )
    app = create_app(
        h.machine,
        h.store,
        h.config,
        secrets,
        h.tmp_path / "frames",
        reference_dir=h.tmp_path / "reference",
        training_dir=h.tmp_path / "training",
    )
    c = TestClient(app)
    c.auth = ("me", PASSWORD)
    return c


GUARDED = (
    "/",
    "/api/status",
    "/api/events",
    "/frame.jpg",
    "/stream.mjpg",
    "/events/1",
    "/api/config",
)


def test_every_route_requires_basic_auth(h: Harness) -> None:
    c = client(h)
    for path in GUARDED:
        r = c.get(path, auth=None)
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Basic")
        assert c.get(path, auth=("me", "wrong")).status_code == 401
        assert c.get(path, auth=("you", PASSWORD)).status_code == 401
    assert c.get("/").status_code == 200
    assert c.get("/healthz", auth=None).json() == {"ok": True}


def test_no_password_configured_locks_everyone_out(h: Harness) -> None:
    secrets = Secrets("", "", "", "", "", "", dash_user="", dash_password="", dash_public_url="")
    c = TestClient(create_app(h.machine, h.store, h.config, secrets, h.tmp_path / "frames"))
    assert c.get("/", auth=("", "")).status_code == 401


def test_mjpeg_yields_each_new_frame_once() -> None:
    frames = FrameBuffer()
    frames.push(Frame(jpeg=b"\xff\xd8one\xff\xd9", at=datetime(2026, 9, 26, 21, 0, 0)))

    async def collect() -> list[bytes]:
        parts = []
        async for part in mjpeg(frames, poll_s=0.01, max_s=0.1):
            parts.append(part)
            if len(parts) == 1:
                frames.push(Frame(jpeg=b"\xff\xd8two\xff\xd9", at=datetime(2026, 9, 26, 21, 0, 1)))
        return parts

    parts = asyncio.run(collect())
    assert len(parts) == 2
    assert parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 7\r\n\r\n")
    assert b"one" in parts[0] and b"two" in parts[1]


def test_status_and_frame(h: Harness) -> None:
    h.load_plates(1)
    h.tick(motion=False)
    c = client(h)
    s = c.get("/api/status").json()
    assert s["state"] == "IDLE"
    assert s["loaded_plates"] == [1]
    assert s["dry_run"] is False
    assert c.get("/frame.jpg").headers["content-type"] == "image/jpeg"


def test_setting_plates_and_config_reach_the_machine(h: Harness) -> None:
    c = client(h)
    h.tick(motion=False)
    assert h.state() == "IDLE"  # nothing marked yet, but armed: it will try the bowls
    r = c.post("/api/plates", json={"loaded": [2, 3]})
    assert r.json()["plates"] == {"1": "empty", "2": "loaded", "3": "loaded"}
    h.tick(motion=False)
    assert h.state() == "IDLE"

    r = c.post(
        "/api/config",
        json={"MIN_GAP_MIN": "45", "DRY_RUN": True, "ACTIVE_START": "20:30"},
    )
    assert r.status_code == 200
    h.tick(motion=False)
    assert h.machine.settings.MIN_GAP_MIN == 45
    assert h.machine.settings.DRY_RUN is True
    assert c.post("/api/config", json={"ACTIVE_END": "nope"}).status_code == 400
    assert c.post("/api/config", json={"BOGUS": 1}).status_code == 400


def test_feed_now_and_event_page(h: Harness) -> None:
    c = client(h)
    h.tick(motion=False)
    c.post("/api/feed-now")
    h.tick(motion=False)
    assert h.state() == "FEEDING"
    events = c.get("/api/events").json()
    open_event = next(e for e in events if e["kind"] == "open")
    page = c.get(f"/events/{open_event['id']}")
    assert page.status_code == 200 and "manual feed" in page.text
    assert c.get(f"/frames/{open_event['frames'][0]}").status_code == 200
    assert c.get("/frames/../config.toml").status_code in (404, 400)


def test_feeder_buttons_enqueue_requests_through_the_machine(h: Harness) -> None:
    c = client(h)
    for action in ("open", "close", "rotate"):
        assert c.post(f"/api/feeder/{action}").status_code == 200
        assert h.store.get(f"manual_{action}_requested")  # flag set for the machine
    assert c.post("/api/feeder/explode").status_code == 400


def test_reference_upload_saves_and_is_counted(h: Harness) -> None:
    c = client(h)
    r = c.post("/api/reference/grrr?filename=test.jpg", content=b"\xff\xd8\xff\xd9jpegbytes")
    assert r.status_code == 200
    assert (h.tmp_path / "reference" / "grrr" / "test.jpg").read_bytes().startswith(b"\xff\xd8")
    assert c.post("/api/reference/dragon?filename=x.jpg", content=b"x").status_code == 400
    assert c.post("/api/reference/grrr?filename=empty.jpg", content=b"").status_code == 400


def test_reference_capture_uses_the_latest_frame(h: Harness) -> None:
    c = client(h)
    assert c.post("/api/reference/cat/capture").status_code == 404  # no frame yet
    h.push_frame()
    r = c.post("/api/reference/cat/capture")
    assert r.status_code == 200
    saved = list((h.tmp_path / "reference" / "cat").glob("cam-*.jpg"))
    assert len(saved) == 1


def test_status_reports_feeder_power(h: Harness) -> None:
    c = client(h)
    s = c.get("/api/status").json()
    assert s["power_cycle_available"] is False
    assert s["plug_off"] is False and s["feeder_booting"] is False and s["feed_retry"] is None
    h.power = object()  # any plug: the button and pills depend on these fields existing
    h.machine = h.new_machine()
    h.store.set("plug_off_at", h.clock.now.isoformat())
    h.store.set("feed_retry", '{"attempts": 1, "trigger": "grrr confirmed", "started_at": "x"}')
    s = client(h).get("/api/status").json()
    assert s["power_cycle_available"] is True
    assert s["plug_off"] is True
    assert s["feed_retry"]["attempts"] == 1


def test_status_reports_lid_and_current_plate(h: Harness) -> None:
    c = client(h)
    h.load_plates(1, 2)
    s0 = c.get("/api/status").json()
    assert s0["lid_open"] is False
    # Manual open (dry run) sets the lid-open flag and records the plate.
    h.config.update({"DRY_RUN": True})
    h.machine.request_manual_action("open")
    h.tick(motion=False)
    s1 = c.get("/api/status").json()
    assert s1["lid_open"] is True
    assert s1["current_plate"] == 1
    # Manual close clears it.
    h.machine.request_manual_action("close")
    h.tick(motion=False)
    assert c.get("/api/status").json()["lid_open"] is False


def test_tailscale_device_gets_in_without_password(h: Harness) -> None:
    c = client(h)
    h.machine.settings = h.config.load()  # DASH_AUTH defaults to "tailscale"
    # On the tailnet, tailscale serve injects this header → no password needed.
    r = c.get("/api/status", auth=None, headers={"Tailscale-User-Login": "pb@example.com"})
    assert r.status_code == 200


def test_tailscale_header_not_trusted_via_cloudflare(h: Harness) -> None:
    c = client(h)
    h.machine.settings = h.config.load()
    # A forged header arriving over the public tunnel (cf-ray present) must NOT grant access.
    r = c.get(
        "/api/status",
        auth=None,
        headers={"Tailscale-User-Login": "pb@example.com", "Cf-Ray": "abc123"},
    )
    assert r.status_code == 401


def test_basic_mode_ignores_tailscale_header(h: Harness) -> None:
    c = client(h)
    h.config.update({"DASH_AUTH": "basic"})
    h.machine.settings = h.config.load()
    assert c.get("/api/status", auth=None, headers={"Tailscale-User-Login": "x"}).status_code == 401
    assert c.get("/api/status").status_code == 200  # client has basic creds


def test_healthz_fails_when_loop_is_stale(h: Harness) -> None:
    c = client(h)
    h.machine.settings = h.config.load()
    assert c.get("/healthz", auth=None).status_code == 200  # no heartbeat yet = starting
    h.store.set("heartbeat_at", h.clock.now.isoformat())
    assert c.get("/healthz", auth=None).status_code == 200  # fresh
    h.clock.advance(minutes=30)
    assert c.get("/healthz", auth=None).status_code == 503  # loop stale → unhealthy → autoheal


def test_photo_list_serve_retag_delete(h: Harness) -> None:
    c = client(h)
    d = h.tmp_path / "training" / "grrr"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"\xff\xd8pic\xff\xd9")
    assert c.get("/api/photos/training/grrr").json()["total"] == 1
    assert c.get("/photos/training/grrr/a.jpg").status_code == 200
    # re-tag grrr -> bowie moves the file
    assert (
        c.post("/api/photos/training/grrr/a.jpg/retag", params={"to": "bowie"}).status_code == 200
    )
    assert not (d / "a.jpg").exists()
    assert (h.tmp_path / "training" / "bowie" / "a.jpg").exists()
    # delete it
    assert c.request("DELETE", "/api/photos/training/bowie/a.jpg").status_code == 200
    assert c.get("/api/photos/training/bowie").json()["total"] == 0
    # unknown set/animal is rejected
    assert c.get("/api/photos/training/dragon").status_code == 404


def test_reset_cycle_route_is_consumed_by_the_machine(h: Harness) -> None:
    c = client(h)
    h.tick(motion=False)
    c.post("/api/feed-now")
    h.tick(motion=False)
    assert h.state() == "FEEDING"
    assert c.post("/api/reset-cycle", auth=None).status_code == 401
    assert c.post("/api/reset-cycle").status_code == 200
    h.tick(motion=False)
    assert h.state() == "IDLE"
    assert "cycle_reset" in [e["kind"] for e in c.get("/api/events").json()]


def test_bedtime_route_starts_settling_and_status_shows_it(h: Harness) -> None:
    c = client(h)
    h.config.update({"FEED_START": "bedtime"})
    h.tick(motion=False)
    assert c.get("/api/status").json()["settle"]["bedtime_at"] is None
    assert c.post("/api/bedtime", auth=None).status_code == 401
    assert c.post("/api/bedtime").status_code == 200
    h.tick(motion=False)
    settle = c.get("/api/status").json()["settle"]
    assert settle["bedtime_at"] and settle["settling"] and settle["settle_min"] == 60
    assert "bedtime" in [e["kind"] for e in c.get("/api/events").json()]


def test_feed_start_rejects_unknown_modes(h: Harness) -> None:
    c = client(h)
    assert c.post("/api/config", json={"FEED_START": "whenever"}).status_code == 400
    assert c.post("/api/config", json={"FEED_START": "bedtime"}).status_code == 200
