"""Process entry point: wires camera, motion, LLM, feeder, store, Slack, the state machine,
and the dashboard together. The machine ticks on a background thread; uvicorn owns the main
thread so signals shut everything down cleanly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .camera import Camera, FrameBuffer
from .config import DEFAULT_CONFIG_PATH, ConfigStore, Secrets
from .dashboard import create_app
from .detect import Identifier
from .events import ReolinkEvents
from .feeder import DryRunFeeder, FeederError, PetlibroCli
from .machine import Machine
from .motion import MotionDetector
from .notify import EmailNotifier, MultiNotifier, PushoverNotifier, SlackNotifier
from .personas import PersonaStore
from .power import build_power_switch
from .ptz import Ptz
from .store import Store
from .themes import ThemeStore
from .vision import ClaudeIdentifier

log = logging.getLogger("animaleyes")

DATA_DIR = Path(os.environ.get("ANIMALEYES_DATA", "data"))
FRONTEND_DIST = Path(os.environ.get("ANIMALEYES_FRONTEND", "frontend/dist"))
TICK_S = 0.5
LOOP_ERROR_REPEAT_S = 600


class UnavailableFeeder:
    """Stands in for the real feeder when PETLIBRO_SERIAL is missing, so DRY_RUN still works."""

    def __init__(self, error: str):
        self.error = error

    def current_plate(self) -> int:
        raise FeederError(self.error)

    def rotate(self) -> None:
        raise FeederError(self.error)

    def open_now(self, plate: int) -> None:
        raise FeederError(self.error)

    def close(self) -> bool:
        raise FeederError(self.error)

    def manual_feed_active(self) -> bool:
        raise FeederError(self.error)


def _yolo(config: ConfigStore):
    from .yolo import YoloConfig, YoloIdentifier

    def settings() -> YoloConfig:
        s = config.load()
        return YoloConfig(
            min_conf=s.YOLO_MIN_CONF,
            grrr_max_box_fraction=s.GRRR_MAX_BOX_FRACTION,
            classifier_path=s.YOLO_CLASSIFIER,
            cls_min_conf=s.YOLO_CLS_MIN_CONF,
        )

    # settings= makes dashboard edits live; without it the knobs were frozen at startup and
    # enabling a classifier silently needed a restart.
    return YoloIdentifier(config.load().YOLO_MODEL, settings(), settings=settings)


def _claude(config: ConfigStore, store: Store, personas: PersonaStore):
    return ClaudeIdentifier(
        DATA_DIR / "reference",
        store,
        model=lambda: config.load().LLM_MODEL,
        personas=personas.load,
    )


def build_identifier(name: str, config: ConfigStore, store: Store, personas: PersonaStore):
    """Select the detector backend by config. Backends share the Identifier interface, so the
    state machine is unchanged whichever is chosen."""
    if name == "yolo":
        log.info("identifier: YOLO (local only)")
        return _yolo(config)
    if name == "cascade":
        from .cascade import CascadeIdentifier

        log.info("identifier: cascade (YOLO gate → Claude confirm)")
        return CascadeIdentifier(
            gate=_yolo(config),
            confirm=_claude(config, store, personas),
            config=config,
            training_dir=DATA_DIR / "training",
        )
    log.info("identifier: Claude (%s)", config.load().LLM_MODEL)
    return _claude(config, store, personas)


def capture_gate(llm, config: ConfigStore):
    """The free, local detector capture mode uses to decide "is this an animal". Reuse the
    active backend's YOLO model if there is one (cascade's gate, or a bare YOLO identifier) so
    we don't load the model twice; otherwise build one. Returns None if YOLO isn't installed
    (capture mode then quietly does nothing — it exists to feed a YOLO dataset)."""
    from .cascade import CascadeIdentifier
    from .yolo import YoloIdentifier

    if isinstance(llm, CascadeIdentifier):
        return llm.gate
    if isinstance(llm, YoloIdentifier):
        return llm
    try:
        return _yolo(config)
    except Exception as exc:  # noqa: BLE001 - absence of a local detector is non-fatal
        log.warning("capture mode: no local detector available (%s)", exc)
        return None


def _healthcheck(url: str):
    """A best-effort GET to a dead-man's-switch URL (e.g. healthchecks.io), or None if unset.
    The external monitor alerts when these stop — the one path that survives the box going down."""
    if not url:
        return None
    import urllib.request

    def ping() -> None:
        with urllib.request.urlopen(url, timeout=10) as resp:
            resp.read()

    return ping


def build(
    secrets: Secrets,
) -> tuple[Machine, Camera, ConfigStore, Store, Identifier, ReolinkEvents | None, PersonaStore]:
    config = ConfigStore(DEFAULT_CONFIG_PATH)
    store = Store(DATA_DIR / "db" / "animaleyes.sqlite")
    s0 = config.load()
    # Seed the persona roster from any descriptions already set in config so dashboard edits
    # aren't lost on the first run with personas.
    personas = PersonaStore(
        DATA_DIR / "personas.json",
        seed={"grrr": s0.GRRR_DESC, "bowie": s0.BOWIE_DESC, "cat": s0.CAT_DESC},
    )
    fps = max(1, s0.CAMERA_FPS)
    frames = FrameBuffer(size=max(12, fps * 4))  # ~4s of history regardless of fps
    motion = MotionDetector()
    camera = Camera(secrets, frames, on_frame=motion.feed, fps=fps)
    llm = build_identifier(s0.IDENTIFIER, config, store, personas)
    events: ReolinkEvents | None = None
    if secrets.kasa_stream_url:
        # Camera-driven motion gate (ONVIF). Keeps the LLM from firing on frame-diff noise.
        events = ReolinkEvents(secrets.kasa_stream_url, secrets.kasa_camera_mac)
    try:
        feeder = PetlibroCli(secrets.petlibro_serial)
    except FeederError as exc:
        log.warning("real feeder unavailable: %s", exc)
        feeder = UnavailableFeeder(str(exc))  # type: ignore[assignment]
    # Slack/log always present (it logs when no webhook); email/SMS added when SMTP is set, so
    # offline/failure alerts reach the phone even with the dashboard closed.
    email = EmailNotifier(
        secrets.smtp_host,
        secrets.smtp_port,
        secrets.smtp_user,
        secrets.smtp_password,
        secrets.smtp_from,
        secrets.alert_recipients(),
    )
    channels = [SlackNotifier(secrets.slack_webhook)]
    if email.enabled:
        channels.append(email)
        log.info("alerts: email/SMS enabled to %s", secrets.alert_recipients())
    pushover = PushoverNotifier(secrets.pushover_token, secrets.pushover_user)
    if pushover.enabled:
        channels.append(pushover)
        log.info("alerts: Pushover emergency alerts enabled")
    notifier = MultiNotifier(channels)
    machine = Machine(
        config=config,
        store=store,
        frames=frames,
        motion=motion,
        llm=llm,
        feeder=feeder,
        dry_feeder=DryRunFeeder(),
        notifier=notifier,
        clock=datetime.now,
        frames_dir=DATA_DIR / "frames",
        dashboard_url=secrets.dash_public_url,
        events=events,
        capture_gate=capture_gate(llm, config),
        capture_dir=DATA_DIR / "training" / "unlabeled",
        healthcheck=_healthcheck(secrets.healthcheck_url),
        power=build_power_switch(
            secrets.smart_plug_kind,
            secrets.smart_plug_host,
            secrets.smart_plug_mac,
            secrets.smart_plug_user,
            secrets.smart_plug_password,
            secrets.smart_plug_off_cmd,
            secrets.smart_plug_on_cmd,
        ),
    )
    return machine, camera, config, store, llm, events, personas


def run_loop(machine: Machine, stop: threading.Event) -> None:
    last_error_at = 0.0
    while not stop.is_set():
        try:
            machine.tick()
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("tick failed")
            if time.monotonic() - last_error_at > LOOP_ERROR_REPEAT_S:
                last_error_at = time.monotonic()
                machine.notifier.alert("animaleyes loop error, see logs")
        stop.wait(TICK_S)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    secrets = Secrets.from_env()
    if not (secrets.dash_user and secrets.dash_password):
        raise SystemExit("DASH_USER and DASH_PASSWORD must be set")
    machine, camera, config, store, llm, events, personas = build(secrets)
    if secrets.kasa_stream_url:
        camera.start()
        if events is not None:
            events.start()
    else:
        log.warning("KASA_STREAM_URL is not set; running without a camera")
    machine.start()
    stop = threading.Event()
    worker = threading.Thread(target=run_loop, args=(machine, stop), name="machine", daemon=True)
    worker.start()
    app = create_app(
        machine,
        store,
        config,
        secrets,
        DATA_DIR / "frames",
        reload_references=llm.reload_references,
        reference_dir=DATA_DIR / "reference",
        training_dir=DATA_DIR / "training",
        personas=personas,
        themes=ThemeStore(DATA_DIR / "themes.json"),
        frontend_dist=FRONTEND_DIST,
        ptz=Ptz(secrets.kasa_stream_url, secrets.kasa_camera_mac)
        if secrets.kasa_stream_url
        else None,
    )
    try:
        uvicorn.run(
            app,
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8081")),
            log_level="warning",
        )
    finally:
        stop.set()
        camera.stop()
        if events is not None:
            events.stop()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
