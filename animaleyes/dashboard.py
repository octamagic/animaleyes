"""Single-page dashboard. No build step; the page polls JSON every 3 seconds and shows the
camera as an MJPEG stream.

Every route except /healthz requires HTTP basic auth (DASH_USER / DASH_PASSWORD). It is
served on loopback only and reached through Cloudflare Tunnel or Tailscale Serve, both of
which terminate TLS, so the password never crosses a network in the clear. Controls write
config.toml or set flags the state machine consumes on its next tick; the dashboard never
talks to the feeder itself.
"""

from __future__ import annotations

import asyncio
import json
import secrets as secrets_lib
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from .camera import FrameBuffer
from .config import ConfigStore, Secrets
from .machine import Machine
from .store import Store

# Module level so FastAPI can resolve the annotation (this file uses postponed annotations).
Credentials = Annotated[
    HTTPBasicCredentials | None, Depends(HTTPBasic(auto_error=False, realm="animaleyes"))
]

STREAM_POLL_S = 0.05  # poll the buffer often enough to pass through higher camera FPS smoothly
# The page reconnects when a stream ends, so a tab left open on a phone cannot hold a
# connection through the tunnel forever.
STREAM_MAX_S = 600

MAIN_EVENT_KINDS = (
    "open",
    "close",
    "veto",
    "sighting",
    "feed_failed",
    "close_failed",
    "rotated_empty_plate",
    "empty_no_food",
    "wanted_food_none_left",
    "startup",
    "bowie_during_feed",
    "cat_during_feed",
    "camera_offline",
    "camera_online",
    "plates_set",
    "grrr_blocked",
    "manual_open",
    "manual_close",
    "manual_close_noop",
    "manual_rotate",
    "manual_open_failed",
    "manual_close_failed",
    "manual_rotate_failed",
    "power_cycle",
    "power_cycle_failed",
    "power_cycle_skipped",
    "power_on",
    "power_on_failed",
    "manual_power_cycle",
    "feeder_online",
    "feeder_not_back",
    "feed_given_up",
    "retry_held",
    "cycle_reset",
)


def create_app(
    machine: Machine,
    store: Store,
    config: ConfigStore,
    secrets: Secrets,
    frames_dir: Path,
    reload_references=None,
    reference_dir: Path | None = None,
    training_dir: Path | None = None,
    personas=None,
    themes=None,
    frontend_dist: Path | None = None,
    ptz=None,
) -> FastAPI:
    app = FastAPI(title="animaleyes", docs_url=None, redoc_url=None)
    index_html = frontend_dist / "index.html" if frontend_dist else None
    PHOTO_SETS = {"reference": reference_dir, "training": training_dir}
    IMG_EXT = (".jpg", ".jpeg", ".png")

    def animal_keys() -> list[str]:
        # The roster is persona-driven; fall back to the fixed trio when no store is wired (tests).
        return personas.keys() if personas is not None else ["grrr", "bowie", "cat"]

    # "unlabeled" is a bucket in the training set only: frames capture mode saved that have no
    # label yet. It's a valid source for listing/viewing/deleting/re-tagging, but never a
    # re-tag *target* (you tag INTO a real animal).
    def _valid_bucket(set_name: str, animal: str) -> bool:
        keys = animal_keys()
        return animal in (keys + ["unlabeled"] if set_name == "training" else keys)

    def photo_dir(set_name: str, animal: str, create: bool = False) -> Path:
        base = PHOTO_SETS.get(set_name)
        if base is None or not _valid_bucket(set_name, animal):
            raise HTTPException(status_code=404, detail="unknown photo set or animal")
        d = base / animal
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def list_photos(set_name: str, animal: str) -> list[str]:
        d = PHOTO_SETS.get(set_name)
        d = (d / animal) if d else None
        if not d or not d.is_dir():
            return []
        files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
        return [p.name for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]

    def ref_counts() -> dict[str, int]:
        # Count the photos actually on disk, independent of the active detector (YOLO doesn't
        # track reference images, so reading them off the identifier wrongly showed 0).
        counts: dict[str, int] = {}
        for a in animal_keys():
            d = (reference_dir / a) if reference_dir else None
            counts[a] = (
                sum(1 for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                if d and d.is_dir()
                else 0
            )
        return counts

    def _basic_ok(credentials: Credentials) -> bool:
        return (
            bool(secrets.dash_password)
            and credentials is not None
            and secrets_lib.compare_digest(
                credentials.username.encode(), secrets.dash_user.encode()
            )
            and secrets_lib.compare_digest(
                credentials.password.encode(), secrets.dash_password.encode()
            )
        )

    def auth(request: Request, credentials: Credentials) -> None:
        mode = machine.settings.DASH_AUTH
        if mode == "none":
            return
        # Tailscale trust: `tailscale serve` injects an identity header for the connecting
        # tailnet device, so being on the tailnet IS the authentication — no password. We only
        # trust it when the request did NOT arrive via the public Cloudflare tunnel (which adds
        # cf-ray), so the header can't be forged from the internet.
        ts_user = request.headers.get("tailscale-user-login")
        via_cloudflare = "cf-ray" in request.headers
        if mode == "tailscale" and ts_user and not via_cloudflare:
            return
        # Basic-auth fallback (public/Cloudflare path, or mode="basic").
        if _basic_ok(credentials):
            return
        raise HTTPException(
            status_code=401,
            detail="login required",
            headers={"WWW-Authenticate": 'Basic realm="animaleyes"'},
        )

    guarded = [Depends(auth)]

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Liveness for the container healthcheck (and autoheal). Fails if the state-machine
        loop has stopped ticking, so a hung-but-running process gets restarted. Unauthenticated
        and reveals nothing sensitive."""
        hb = store.get("heartbeat_at")
        if hb:
            age = (machine.clock() - datetime.fromisoformat(hb)).total_seconds()
            limit = max(180, 3 * machine.settings.HEARTBEAT_MIN * 60)
            if age > limit:
                raise HTTPException(status_code=503, detail=f"loop stale {int(age)}s")
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse, dependencies=guarded)
    def index() -> HTMLResponse:
        # Serve the built React app; fall back to a minimal page when it isn't built (tests).
        html = index_html.read_text() if (index_html and index_html.is_file()) else FALLBACK_PAGE
        # Never cache the shell — the hashed assets it points at change on every build, so a
        # cached old index.html would reference assets that no longer exist (blank page).
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})

    @app.get("/api/status", dependencies=guarded)
    def status() -> dict[str, Any]:
        now = machine.clock()
        settings = machine.settings
        heartbeat = store.get("heartbeat_at")
        last_frame = machine.frames.last_frame_at()
        calls, cost = store.llm_totals_since(now.replace(hour=0, minute=0, second=0, microsecond=0))
        next_feed = machine.next_allowed_feed()
        verdict = machine.last_verdict.to_dict() if machine.last_verdict else None
        return {
            "now": now.isoformat(timespec="seconds"),
            "state": machine.state,
            "enabled": settings.ENABLED,
            "dry_run": settings.DRY_RUN,
            "active_window": f"{settings.ACTIVE_START}-{settings.ACTIVE_END}",
            "in_window": settings.is_active_at(now.time()),
            "plates": store.plates(),
            "loaded_plates": store.loaded_plates(),
            "plates_tried": machine.plates_tried(),
            "feeds_this_window": store.get_int("feeds_this_window"),
            "last_open_at": store.get("last_open_at"),
            "motion_source": settings.MOTION_SOURCE,
            "camera_motion": getattr(machine.events, "motion_state", None),
            "camera_animal": getattr(machine.events, "animal_state", None),
            "camera_events_ok": (machine.events.last_error is None)
            if machine.events is not None
            else None,
            # Prefer the feeder's real lid state (catches opens done from the PetLibro app);
            # fall back to what the machine knows from its own actions.
            "lid_open": (store.get("lid_actual_open") == "1")
            if store.get("lid_actual_open") is not None
            else (
                machine.state in ("OPENING", "VERIFYING", "FEEDING", "CLOSING")
                or bool(store.get("lid_manual_open"))
            ),
            "current_plate": store.get_int("current_plate", 0) or None,
            "feeding_plate": (store.get_int("feeding_plate", 0) or None)
            if machine.state == "FEEDING"
            else None,
            "next_allowed_feed": next_feed.isoformat(timespec="seconds") if next_feed else None,
            "next_allowed_feed_in_s": max(0, int((next_feed - now).total_seconds()))
            if next_feed
            else 0,
            "heartbeat_age_s": _age(heartbeat, now),
            "camera_age_s": int((now - last_frame).total_seconds()) if last_frame else None,
            "camera_offline": machine.camera_offline,
            "motion_fraction": round(machine.motion.last_fraction, 4),
            "llm_calls_today": calls,
            "llm_cost_today_usd": round(cost, 4),
            "llm_model": settings.LLM_MODEL,
            "identifier": settings.IDENTIFIER,
            "ptz_available": ptz is not None,
            # Feeder power (smart plug): drives the Power cycle button and the rebooting pills.
            "power_cycle_available": machine.power is not None,
            "plug_off": machine.plug_is_off(),
            "plug_cycled_at": store.get("plug_cycled_at"),
            "feeder_booting": machine.feeder_booting(),
            "feed_retry": machine._feed_retry(),
            "last_verdict": verdict,
            "last_verdict_age_s": int((now - machine.last_verdict_at).total_seconds())
            if machine.last_verdict_at
            else None,
            "last_llm_at": machine.last_llm_at.isoformat(timespec="seconds")
            if machine.last_llm_at
            else None,
            "reference_counts": ref_counts(),
            "training_counts": {
                a: len(list_photos("training", a)) for a in [*animal_keys(), "unlabeled"]
            },
            "personas": [p.to_dict() for p in personas.load()] if personas is not None else [],
            "capture_mode": settings.CAPTURE_MODE,
            "feeding_since": store.get("opened_at") if machine.state == "FEEDING" else None,
        }

    @app.get("/api/config", dependencies=guarded)
    def get_config() -> dict[str, Any]:
        return config.load().__dict__

    @app.post("/api/config", dependencies=guarded)
    async def set_config(request: Request) -> dict[str, Any]:
        changes = await request.json()
        try:
            settings = config.update(changes)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.add_event(machine.clock(), "config", "dashboard", changes)
        return settings.__dict__

    @app.post("/api/plates", dependencies=guarded)
    async def set_plates(request: Request) -> dict[str, Any]:
        body = await request.json()
        loaded = {int(p) for p in body.get("loaded", [])}
        now = machine.clock()
        for plate in (1, 2, 3):
            store.set_plate(plate, "loaded" if plate in loaded else "empty", now)
        store.set("plates_updated", 1)
        return {"plates": store.plates()}

    # --- camera pan/tilt (PTZ over ONVIF; moves the camera, not the feeder) ----------
    PTZ_DIRS = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}

    @app.post("/api/ptz/nudge", dependencies=guarded)
    def ptz_nudge(dir: str) -> dict[str, str]:
        if ptz is None:
            raise HTTPException(status_code=503, detail="no camera")
        if dir not in PTZ_DIRS:
            raise HTTPException(status_code=400, detail="bad direction")
        s = machine.settings
        dx, dy = PTZ_DIRS[dir]
        try:
            ptz.nudge(dx * s.PTZ_SPEED, dy * s.PTZ_SPEED, s.PTZ_STEP_MS)
        except Exception as exc:  # noqa: BLE001 - surface camera errors to the UI
            raise HTTPException(status_code=502, detail=f"ptz failed: {exc}") from exc
        return {"ok": dir}

    @app.post("/api/ptz/stop", dependencies=guarded)
    def ptz_stop() -> dict[str, bool]:
        if ptz is not None:
            try:
                ptz.stop()
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True}

    @app.get("/api/ptz/presets", dependencies=guarded)
    def ptz_presets() -> dict[str, Any]:
        if ptz is None:
            return {"presets": []}
        try:
            return {"presets": ptz.presets()}
        except Exception:  # noqa: BLE001
            return {"presets": []}

    @app.post("/api/ptz/preset", dependencies=guarded)
    async def ptz_set_preset(request: Request) -> dict[str, bool]:
        if ptz is None:
            raise HTTPException(status_code=503, detail="no camera")
        name = (await request.json()).get("name", "preset")
        ptz.set_preset(str(name)[:40])
        return {"ok": True}

    @app.post("/api/ptz/goto", dependencies=guarded)
    def ptz_goto(token: str) -> dict[str, bool]:
        if ptz is None:
            raise HTTPException(status_code=503, detail="no camera")
        ptz.goto(token)
        return {"ok": True}

    @app.post("/api/feed-now", dependencies=guarded)
    def feed_now() -> dict[str, str]:
        machine.request_feed_now()
        return {"ok": "feed requested; the state machine opens on its next tick"}

    @app.post("/api/reset-cycle", dependencies=guarded)
    def reset_cycle() -> dict[str, str]:
        machine.request_reset_cycle()
        return {"ok": "reset requested; the state machine applies it on its next tick"}

    @app.post("/api/feeder/{action}", dependencies=guarded)
    def feeder_action(action: str) -> dict[str, str]:
        # The page only enqueues the request; the state machine is still the sole caller of
        # the feeder and performs it on its next tick (product invariant).
        try:
            machine.request_manual_action(action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": f"{action} requested; the state machine runs it on its next tick"}

    def _animal_dir(animal: str) -> Path:
        if animal not in animal_keys() or reference_dir is None:
            raise HTTPException(status_code=400, detail="unknown animal")
        d = reference_dir / animal
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_reference(animal: str, name: str, jpeg: bytes) -> dict[str, Any]:
        safe = Path(name).name or "frame.jpg"
        if not safe.lower().endswith((".jpg", ".jpeg", ".png")):
            safe += ".jpg"
        (_animal_dir(animal) / safe).write_bytes(jpeg)
        if reload_references:
            reload_references()
        return {"saved": safe, "reference_counts": ref_counts()}

    @app.post("/api/reference/{animal}", dependencies=guarded)
    async def upload_reference(animal: str, request: Request) -> dict[str, Any]:
        # Raw image bytes in the body (no multipart dependency); filename in the query.
        name = request.query_params.get("filename", "upload.jpg")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty upload")
        return _save_reference(animal, name, body)

    @app.post("/api/reference/{animal}/capture", dependencies=guarded)
    def capture_reference(animal: str) -> dict[str, Any]:
        frames = machine.frames.latest(1)
        if not frames:
            raise HTTPException(status_code=404, detail="no camera frame yet")
        name = f"cam-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
        return _save_reference(animal, name, frames[0].jpeg)

    @app.post("/api/reload-references", dependencies=guarded)
    def reload_refs() -> dict[str, Any]:
        if reload_references:
            reload_references()
        return {"reference_counts": ref_counts()}

    # --- photo management (reference + auto-collected training frames) ---------------
    @app.get("/api/photos/{set_name}/{animal}", dependencies=guarded)
    def photos_list(set_name: str, animal: str, limit: int = 60) -> dict[str, Any]:
        photo_dir(set_name, animal)  # validates set + animal (404 otherwise)
        names = list_photos(set_name, animal)
        return {"total": len(names), "files": names[:limit]}

    @app.get("/photos/{set_name}/{animal}/{name}", dependencies=guarded)
    def photo_file(set_name: str, animal: str, name: str) -> FileResponse:
        d = photo_dir(set_name, animal)
        path = (d / Path(name).name).resolve()
        if not path.is_file() or d.resolve() not in path.parents:
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.delete("/api/photos/{set_name}/{animal}/{name}", dependencies=guarded)
    def photo_delete(set_name: str, animal: str, name: str) -> dict[str, bool]:
        d = photo_dir(set_name, animal)
        path = (d / Path(name).name).resolve()
        if path.is_file() and d.resolve() in path.parents:
            path.unlink()
        if set_name == "reference" and reload_references:
            reload_references()
        return {"ok": True}

    @app.post("/api/photos/{set_name}/{animal}/{name}/retag", dependencies=guarded)
    def photo_retag(set_name: str, animal: str, name: str, to: str) -> dict[str, bool]:
        if to not in animal_keys():  # you tag INTO a real animal, never back to "unlabeled"
            raise HTTPException(status_code=400, detail="can only re-tag to a known animal")
        src_dir = photo_dir(set_name, animal)
        src = (src_dir / Path(name).name).resolve()
        if not src.is_file() or src_dir.resolve() not in src.parents:
            raise HTTPException(status_code=404)
        dst = photo_dir(set_name, to, create=True) / src.name
        src.rename(dst)
        if set_name == "reference" and reload_references:
            reload_references()
        return {"ok": True}

    # --- animal personas (roster the detector knows about) ----------------------------
    @app.get("/api/personas", dependencies=guarded)
    def personas_list() -> list[dict[str, Any]]:
        return [p.to_dict() for p in personas.load()] if personas is not None else []

    @app.post("/api/personas", dependencies=guarded)
    async def personas_upsert(request: Request) -> dict[str, Any]:
        if personas is None:
            raise HTTPException(status_code=503, detail="personas unavailable")
        body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            persona = personas.upsert(name, str(body.get("description", "")), body.get("key"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such persona {exc}") from exc
        # Make the photo folders so uploads/capture have somewhere to land, and rebuild the
        # detector's cached reference prefix + schema so a new/renamed animal takes effect.
        for base in (reference_dir, training_dir):
            if base is not None:
                (base / persona.key).mkdir(parents=True, exist_ok=True)
        if reload_references:
            reload_references()
        store.add_event(machine.clock(), "persona", f"saved {persona.name}", persona.to_dict())
        return persona.to_dict()

    # --- custom colour themes (presentation only) -------------------------------------
    @app.get("/api/themes", dependencies=guarded)
    def themes_list() -> list[dict[str, Any]]:
        return [t.to_dict() for t in themes.load()] if themes is not None else []

    @app.post("/api/themes", dependencies=guarded)
    async def themes_upsert(request: Request) -> dict[str, Any]:
        if themes is None:
            raise HTTPException(status_code=503, detail="themes unavailable")
        body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            theme = themes.upsert(name, body.get("colors", {}), body.get("id"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such theme {exc}") from exc
        return theme.to_dict()

    @app.delete("/api/themes/{theme_id}", dependencies=guarded)
    def themes_delete(theme_id: str) -> dict[str, bool]:
        if themes is None:
            raise HTTPException(status_code=503, detail="themes unavailable")
        themes.delete(theme_id)
        return {"ok": True}

    @app.delete("/api/personas/{key}", dependencies=guarded)
    def personas_delete(key: str) -> dict[str, bool]:
        if personas is None:
            raise HTTPException(status_code=503, detail="personas unavailable")
        try:
            personas.delete(key)
        except ValueError as exc:  # the feedable target can't be removed (invariant)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Remove its photos too so a deleted animal doesn't linger in the galleries.
        for base in (reference_dir, training_dir):
            d = (base / key) if base else None
            if d and d.is_dir():
                for p in d.iterdir():
                    if p.is_file():
                        p.unlink()
                d.rmdir()
        if reload_references:
            reload_references()
        store.add_event(machine.clock(), "persona", f"deleted {key}", {"key": key})
        return {"ok": True}

    @app.get("/frame.jpg", dependencies=guarded)
    def frame() -> Response:
        frames = machine.frames.latest(1)
        if not frames:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(
            content=frames[0].jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.get("/stream.mjpg", dependencies=guarded)
    def stream() -> StreamingResponse:
        # Read-only view of the frames the camera thread already decodes; it opens no new
        # connection to the camera. POC: runs at the ingest rate (~2 fps), which motion
        # detection is tuned for.
        return StreamingResponse(
            mjpeg(machine.frames),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/events", dependencies=guarded)
    def events(all: bool = False, limit: int = 100) -> JSONResponse:
        rows = store.events(limit=limit, kinds=None if all else MAIN_EVENT_KINDS)
        return JSONResponse([_event_json(e) for e in rows])

    @app.get("/events/{event_id}", response_class=HTMLResponse, dependencies=guarded)
    def event_page(event_id: int) -> str:
        event = store.event(event_id)
        if not event:
            raise HTTPException(status_code=404)
        images = "".join(f'<img src="/frames/{name}" alt="{name}">' for name in event.frames)
        return EVENT_PAGE.format(
            id=event.id,
            at=event.at.strftime("%Y-%m-%d %H:%M:%S"),
            kind=event.kind,
            reason=_escape(event.reason),
            data=_escape(json.dumps(event.data, indent=2)),
            images=images,
        )

    @app.get("/frames/{name}", dependencies=guarded)
    def frame_file(name: str) -> FileResponse:
        path = (frames_dir / Path(name).name).resolve()
        if not path.is_file() or frames_dir.resolve() not in path.parents:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    # Static assets for the built React app (bundles, no secrets → unauthenticated).
    if frontend_dist and (frontend_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    return app


MJPEG_BOUNDARY = "frame"


async def mjpeg(
    frames: FrameBuffer, poll_s: float = STREAM_POLL_S, max_s: float = STREAM_MAX_S
) -> AsyncIterator[bytes]:
    """Yield each new frame in the buffer as one multipart/x-mixed-replace part."""
    deadline = time.monotonic() + max_s
    last_at: datetime | None = None
    while time.monotonic() < deadline:
        latest = frames.latest(1)
        if latest and latest[0].at != last_at:
            last_at = latest[0].at
            jpeg = latest[0].jpeg
            yield (
                f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg
                + b"\r\n"
            )
        await asyncio.sleep(poll_s)


def _age(iso: str | None, now: datetime) -> int | None:
    if not iso:
        return None
    return int((now - datetime.fromisoformat(iso)).total_seconds())


def _event_json(e) -> dict[str, Any]:
    return {
        "id": e.id,
        "at": e.at.isoformat(timespec="seconds"),
        "kind": e.kind,
        "reason": e.reason,
        "data": e.data,
        "frames": e.frames,
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def seconds_to_text(seconds: int) -> str:
    return str(timedelta(seconds=seconds))


FALLBACK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>animaleyes</title></head>
<body style="background:#12150e;color:#ebecd7;font:16px system-ui;padding:24px">
<h1>🐾 animaleyes</h1><p>The dashboard UI was not built into this image.</p>
<p>Build with <code>docker compose up -d --build</code> (the frontend build stage compiles the React app).</p>
<p>The API is up: <a style="color:#93c0a4" href="/api/status">/api/status</a></p>
</body></html>"""

EVENT_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>event {id}</title>
<style>
 :root{{--bg:#0e0f13;--surface:#171920;--border:#2a2e3a;--fg:#e7e9ef;--muted:#949aa7;--accent:#6aa9ff}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
 .wrap{{max-width:760px;margin:0 auto;padding:16px}}
 a{{color:var(--accent);text-decoration:none}}
 h2{{margin:10px 0 2px;font-size:20px}}
 .meta{{color:var(--muted);font-size:13px;margin-bottom:14px}}
 .reason{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;margin:12px 0}}
 img{{width:100%;border-radius:12px;margin:8px 0;border:1px solid var(--border)}}
 details{{margin-top:12px}} summary{{color:var(--muted);cursor:pointer;font-size:13px}}
 pre{{background:var(--surface);border:1px solid var(--border);padding:12px;border-radius:12px;
   white-space:pre-wrap;word-break:break-word;font-size:12px;color:var(--muted)}}
</style></head><body><div class="wrap">
<a href="/">&larr; Back to dashboard</a>
<h2>{kind}</h2>
<div class="meta">event #{id} · {at}</div>
<div class="reason">{reason}</div>
{images}
<details><summary>Raw data</summary><pre>{data}</pre></details>
</div></body></html>"""
