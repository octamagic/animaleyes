"""Runtime settings from config.toml plus secrets from the environment.

Settings are re-read on every loop so dashboard edits take effect without a restart.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import asdict, dataclass, fields
from datetime import time
from pathlib import Path

import tomli_w

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(os.environ.get("ANIMALEYES_CONFIG", "config.toml"))
FEED_STARTS = ("schedule", "bedtime")


@dataclass
class Settings:
    ENABLED: bool = True
    DRY_RUN: bool = True
    ACTIVE_START: str = "21:00"
    ACTIVE_END: str = "06:00"
    MIN_GAP_MIN: int = 60
    # When the night's feeding may start. "schedule": any time in the active window. "bedtime":
    # only after "We're in bed" is pressed on the dashboard, and the first feed after that waits
    # until Grrr has gone SETTLE_MIN without a confident sighting (asleep) and then comes back.
    # Circling the feeder right after bed keeps restarting that clock. Later feeds that night
    # follow the usual rules.
    FEED_START: str = "schedule"
    SETTLE_MIN: int = 60
    # Bedtime mode: Slack nudge once the window has been open this long with no "We're in bed"
    # (0 = never). Without the press nothing is fed that night.
    BEDTIME_REMIND_MIN: int = 150
    # If nobody marked a bowl as loaded for this window, still feed: when Grrr is confirmed,
    # open whatever bowl is under the lid, check it for food (VERIFY_FOOD), and move on to the
    # next untried bowl if it's empty, until every bowl has been tried once. Bowls marked from
    # the dashboard are still honoured exactly as marked.
    FEED_WITHOUT_LOADED_PLATES: bool = True
    LEAVE_TIMEOUT_S: int = 120
    # Close as soon as the bowl reads empty and she's away from it, this many polls in a row
    # (FEEDING_POLL_S apart), instead of waiting out LEAVE_TIMEOUT_S.
    CLOSE_WHEN_EMPTY: bool = True
    EATEN_CONFIRMATIONS: int = 2
    FEEDING_MAX_MIN: int = 15
    FEEDING_POLL_S: int = 20
    CONFIRMATIONS_REQUIRED: int = 3
    # Confident Grrr reads count toward confirmation for this long; "none"/"unsure" reads in
    # between don't reset the count (she's just out of frame for a moment), another animal does.
    CONFIRM_WINDOW_S: int = 60
    GRRR_MIN_CONF: float = 0.65  # Grrr reads ~0.72-0.9 in IR; 0.8 was a touch high
    MOTION_PIXEL_FRACTION: float = 0.02
    MOTION_HOLD_S: int = 20
    MOTION_SOURCE: str = "camera"  # "camera" = ONVIF motion events; "frames" = frame-diff
    CAMERA_FPS: int = 8  # frames/sec ffmpeg pulls; higher = smoother live view (restart to apply)
    FEED_RETRY_BACKOFF_S: int = 120  # after a failed open, wait before trying to open again
    # A failed open is retried this many more times (after a power cycle when a plug is set,
    # else after FEED_RETRY_BACKOFF_S) before the machine gives up for that approach.
    FEED_RETRIES: int = 3
    # Smart plug on the feeder's power (SMART_PLUG_* in .env). Any error from a live feeder
    # command pulls the plug at once, off for POWER_CYCLE_OFF_S. Power back on, the machine waits
    # POWER_CYCLE_BOOT_S, then asks the cloud every POWER_CYCLE_PROBE_S until the feeder has
    # checked in since the power-on and reports no fault; after POWER_CYCLE_BOOT_MAX_S it stops
    # waiting and proceeds regardless. Cycles outside a feed attempt are capped to one per
    # POWER_CYCLE_MIN_GAP_MIN; a feed attempt's cycles are capped by FEED_RETRIES instead.
    POWER_CYCLE_ON_FAULT: bool = True
    POWER_CYCLE_OFF_S: int = 10
    POWER_CYCLE_BOOT_S: int = 45
    POWER_CYCLE_PROBE_S: int = 15
    POWER_CYCLE_BOOT_MAX_S: int = 300
    POWER_CYCLE_MIN_GAP_MIN: int = 10
    LID_POLL_S: int = 20  # how often to read the feeder's real lid state (0 = never)
    # Dashboard auth: "tailscale" = trust any device on the tailnet (no password), basic-auth
    # fallback off-tailnet; "basic" = always require the password; "none" = open (don't).
    DASH_AUTH: str = "tailscale"
    PTZ_SPEED: float = 0.4  # pan/tilt velocity 0..1 for a nudge
    PTZ_STEP_MS: int = 500  # how long one tap moves the camera
    LLM_MIN_INTERVAL_S: int = 3
    HEARTBEAT_MIN: int = 5
    LLM_MODEL: str = "claude-opus-5"
    # "cascade" = local YOLO gate (dog/cat/none, free) that only escalates to Claude when a
    # dog is present (accurate + cheap); "claude" = always Claude; "yolo" = local only.
    IDENTIFIER: str = "cascade"
    # Cascade only: when the free gate sees nothing during motion it may be blind (a dog filling
    # the frame isn't a "dog" to it), so Claude gets a look this often. 0 = never (cheapest, but
    # close-up visits can go unanswered). After Claude sees an animal the gate missed, it keeps
    # answering for CASCADE_STICKY_S so confirmations can complete.
    CASCADE_PROBE_S: int = 10
    CASCADE_STICKY_S: int = 20
    YOLO_MODEL: str = "yolo11n.pt"  # pretrained COCO; dog/cat/person
    YOLO_MIN_CONF: float = 0.45
    # Identity classifier trained from the tagged frames (tools/train_classifier.py). Empty =
    # the size rule (big box = Bowie), which is wrong whenever Grrr is close to the camera.
    YOLO_CLASSIFIER: str = ""  # e.g. data/models/animals-cls.pt
    YOLO_CLS_MIN_CONF: float = 0.6  # below this a dog is "unsure" (never fed; cascade asks Claude)
    GRRR_MAX_BOX_FRACTION: float = 0.5  # YOLO-only mode: dog box this fraction or smaller = Grrr
    # The open decision needs Grrr confirmed at the bowl. The bowl is closed/empty before a
    # feed, so she hovers near it rather than head-in-bowl; requiring at_bowl blocks nearly
    # every feed, so default off (presence is enough; identity still has to be Grrr).
    OPEN_REQUIRES_AT_BOWL: bool = False
    # Auto-collect labelled IR crops for a future local classifier: when the cascade's Claude
    # step confidently says Grrr/Bowie, save that frame under data/training/<animal>/.
    COLLECT_TRAINING: bool = True
    TRAINING_MIN_CONF: float = 0.6
    TRAINING_MIN_GAP_S: int = 5  # don't save more than one crop this often (avoid bursts)
    # Capture mode: a data-gathering mode (independent of feeding). When on, any motion the
    # free local detector reads as an animal is saved, unlabelled, to data/training/unlabeled/
    # for the human to tag. Runs all day regardless of the active window; never touches the
    # feeder (it only writes image files).
    CAPTURE_MODE: bool = False
    CAPTURE_MIN_GAP_S: int = 3  # at most one captured frame this often
    # Always-on watch: the free local detector checks the bowl every WATCH_INTERVAL_S in every
    # state where the feeding machine isn't already looking (off hours included) and logs who
    # it sees. Logging and the dashboard only; it never opens the feeder.
    ALWAYS_WATCH: bool = True
    WATCH_INTERVAL_S: int = 2
    # Feeding verification: after opening a plate, check with the camera that the served bowl
    # actually has food. If it's confirmed empty, close it and rotate to the next loaded plate,
    # so a wrong/empty plate doesn't mean a midnight rescue. Off by default (it moves the tray
    # based on a vision read, so enable it deliberately once the close-up framing is dialled in).
    VERIFY_FOOD: bool = False
    VERIFY_POLL_S: int = 10  # how often to run the food check while verifying
    VERIFY_EMPTY_CONFIRMATIONS: int = 2  # consecutive "empty" reads before rotating (anti-misfire)
    VERIFY_TIMEOUT_S: int = 90  # if still unsure after this, stop second-guessing and just feed
    MAX_ROTATE_FOR_FOOD: int = 2  # cap rotations hunting for food (never churn the whole tray)
    # Per-animal descriptions fed to Claude (editable from the dashboard). These label the
    # reference photos and give the model distinguishing cues; the IR/size guidance is in the
    # system prompt. Keep "only Grrr is fed" explicit.
    GRRR_DESC: str = (
        "Grrr: a tiny, old, black schnoodle. Small body, short legs, scruffy/curly coat. "
        "The ONLY animal that may be fed."
    )
    BOWIE_DESC: str = (
        "Bowie: a medium-sized, tan/blonde, lanky dog with a smooth short coat. Much larger "
        "and longer-legged than Grrr. Never fed."
    )
    CAT_DESC: str = (
        "The cat: sleek short coat, upright triangular ears, short muzzle, long tail. Never fed."
    )

    def active_window(self) -> tuple[time, time]:
        return parse_hhmm(self.ACTIVE_START), parse_hhmm(self.ACTIVE_END)

    def is_active_at(self, t: time) -> bool:
        """True when `t` is inside [ACTIVE_START, ACTIVE_END], which may cross midnight."""
        start, end = self.active_window()
        if start <= end:
            return start <= t < end
        return t >= start or t < end


def parse_hhmm(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def coerce(settings: Settings, key: str, raw: object) -> object:
    """Convert a dashboard form value to the field's declared type."""
    field_type = {f.name: f.type for f in fields(settings)}[key]
    if field_type in ("bool", bool):
        return str(raw).lower() in ("1", "true", "on", "yes")
    if field_type in ("int", int):
        return int(raw)  # type: ignore[call-overload]
    if field_type in ("float", float):
        return float(raw)  # type: ignore[arg-type]
    value = str(raw)
    if key in ("ACTIVE_START", "ACTIVE_END"):
        parse_hhmm(value)  # validate
    if key == "FEED_START" and value not in FEED_STARTS:
        raise ValueError(f"FEED_START must be one of {FEED_STARTS}")
    return value


class ConfigStore:
    """Loads and saves config.toml. Unknown keys are preserved on save."""

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH):
        self.path = path

    def load(self) -> Settings:
        if not self.path.exists():
            return Settings()
        known = {f.name for f in fields(Settings)}
        try:
            data = tomllib.loads(self.path.read_text())
            settings = Settings(**{k: v for k, v in data.items() if k in known})
            self._last_good = settings
            return settings
        except Exception as exc:
            # A malformed config.toml (e.g. a hand/dashboard edit with a syntax error) must
            # never crash-loop the daemon. Fall back to the last good settings, else defaults.
            log.error("config.toml invalid (%s); using last-good/defaults", exc)
            return getattr(self, "_last_good", None) or Settings()

    def update(self, changes: dict[str, object]) -> Settings:
        current = self.load()
        for key, raw in changes.items():
            if key not in {f.name for f in fields(Settings)}:
                raise KeyError(key)
            setattr(current, key, coerce(current, key, raw))
        self.path.write_text(tomli_w.dumps(asdict(current)))
        return current


@dataclass(frozen=True)
class Secrets:
    kasa_stream_url: str
    kasa_email: str
    kasa_password: str
    anthropic_api_key: str
    petlibro_serial: str
    slack_webhook: str
    dash_user: str
    dash_password: str
    dash_public_url: str
    kasa_camera_mac: str = ""
    # Email/SMS alerts (optional). SMS = email a carrier gateway address (see .env.example).
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    alert_to: str = ""  # comma-separated email and/or carrier-SMS addresses
    pushover_token: str = ""  # Pushover app token (emergency-priority wake-me alerts)
    pushover_user: str = ""  # Pushover user/group key
    healthcheck_url: str = ""  # dead-man's-switch: pinged each heartbeat; alerts if pings stop
    # Smart plug on the feeder's power: a Kasa/Tapo plug's LAN address (+ Tapo credentials),
    # or a pair of shell commands for any other kind of plug.
    smart_plug_kind: str = "shelly"  # shelly | kasa (or set the *_CMD pair for anything else)
    smart_plug_host: str = ""
    smart_plug_mac: str = ""
    smart_plug_user: str = ""
    smart_plug_password: str = ""
    smart_plug_off_cmd: str = ""
    smart_plug_on_cmd: str = ""

    @classmethod
    def from_env(cls) -> Secrets:
        env = os.environ.get
        return cls(
            kasa_stream_url=env("KASA_STREAM_URL", ""),
            kasa_email=env("KASA_EMAIL", ""),
            kasa_password=env("KASA_PASSWORD", ""),
            anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
            petlibro_serial=env("PETLIBRO_SERIAL", ""),
            slack_webhook=env("SLACK_WEBHOOK", ""),
            dash_user=env("DASH_USER", ""),
            dash_password=env("DASH_PASSWORD", ""),
            dash_public_url=env("DASH_PUBLIC_URL", "http://localhost:8081").rstrip("/"),
            kasa_camera_mac=env("KASA_CAMERA_MAC", ""),
            smtp_host=env("SMTP_HOST", ""),
            smtp_port=int(env("SMTP_PORT", "587") or "587"),
            smtp_user=env("SMTP_USER", ""),
            smtp_password=env("SMTP_PASSWORD", ""),
            smtp_from=env("SMTP_FROM", ""),
            alert_to=env("ALERT_TO", ""),
            pushover_token=env("PUSHOVER_TOKEN", ""),
            pushover_user=env("PUSHOVER_USER", ""),
            healthcheck_url=env("HEALTHCHECK_URL", ""),
            smart_plug_kind=env("SMART_PLUG_KIND", "shelly"),
            smart_plug_host=env("SMART_PLUG_HOST", ""),
            smart_plug_mac=env("SMART_PLUG_MAC", ""),
            smart_plug_user=env("SMART_PLUG_USER", ""),
            smart_plug_password=env("SMART_PLUG_PASSWORD", ""),
            smart_plug_off_cmd=env("SMART_PLUG_OFF_CMD", ""),
            smart_plug_on_cmd=env("SMART_PLUG_ON_CMD", ""),
        )

    def alert_recipients(self) -> list[str]:
        return [r.strip() for r in self.alert_to.split(",") if r.strip()]
