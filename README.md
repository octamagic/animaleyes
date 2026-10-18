# animaleyes

Watches a TP-Link Kasa camera mounted over a PetLibro Polar (PLAF109) wet-food feeder and
opens the feeder **only for Grrr**, a tiny old black schnoodle. Bowie (medium blonde lanky
dog) and the cat are never fed and get a veto event instead.

**This is a POC.** Identification is done by sending camera frames to Claude with cached
reference photos. It is built to be running and monitorable from a phone in one day, with a
codebase that can grow into the real system. Corners cut on purpose are marked `POC` in
comments and listed at the bottom.

## How it works

```
Kasa camera ──ffmpeg (2 fps JPEG)──▶ FrameBuffer ──▶ MotionDetector
                                         │                │
                                         ▼                ▼
                                   Claude vision ◀── state machine ──▶ petlibro-cli ──▶ feeder
                                   (ref photos cached)    │
                                                          ├──▶ SQLite (state, plates, events, LLM log)
                                                          ├──▶ Slack webhook
                                                          └──▶ dashboard 127.0.0.1:8081 (live MJPEG)
```

States: `OUTSIDE_WINDOW → IDLE → WATCHING → OPENING → FEEDING → CLOSING → COOLDOWN → IDLE|DONE`.
The machine knows which of the three plates are loaded, rotates the tray to a loaded plate
before opening, and after Grrr leaves asks the model whether the bowl was eaten. An eaten
bowl advances to the next plate on her next visit; an untouched bowl is offered again.
See `animaleyes/machine.py` for the exact rules and `tests/test_machine.py` for the
scenarios that are locked in (two-chamber night, min gap, leave timeout, veto during feed,
restart mid-session cannot double-feed).

## Run it

```bash
cp .env.example .env            # fill in camera, Anthropic, PetLibro, Slack, DASH_USER/PASSWORD
mkdir -p config && cp config.toml config/config.toml
docker compose up -d --build
docker compose logs -f
tools/expose.sh grrr.threaditate.com   # once: Cloudflare hostname + Tailscale Serve
```

**Camera address.** The camera takes its IP from DHCP, so the URL in `.env` uses a
`{host}` placeholder and `KASA_CAMERA_MAC` identifies the camera. On every (re)connect the
app looks the MAC up in the ARP table, knocking on the stream port across the /24 first if
the entry has aged out. That is why the container runs with `network_mode: host`. A DHCP
reservation on the router is a good belt-and-braces addition but not required.

`DRY_RUN` is `true` by default: the machine logs "would open plate N" and never touches
the feeder. Watch a full dry night, then flip `DRY_RUN` off on the dashboard.

**PetLibro login.** The container needs a cached token. The first `feed` call logs in
automatically when the token is missing or expired, which signs the PetLibro phone app out
of that account. Use a second PetLibro account with the feeder shared to it (set
`PETLIBRO_EMAIL`/`PETLIBRO_PASSWORD` to that account) and this never matters.

Local development without Docker:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" -e ./petlibro-cli
brew install ffmpeg          # or apt install ffmpeg
.venv/bin/animaleyes         # dashboard on http://127.0.0.1:8081/ (basic auth)
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Reference photos

Put several photos of each animal in `data/reference/grrr/`, `data/reference/bowie/`,
`data/reference/cat/` (jpg/png; they are downscaled to 640 px before sending). Include
night IR frames from this camera, since that is what the model will actually see. Grab
candidates from the real camera with:

```bash
.venv/bin/python tools/capture.py --every 10 --minutes 60     # -> data/captures/
.venv/bin/python tools/capture.py --once                       # one frame, sanity check
```

Then press **Reload reference photos** on the dashboard (or restart). To tune prompts and
thresholds without waiting for the animals:

```bash
.venv/bin/python tools/replay.py data/captures            # identify, 3-frame windows
.venv/bin/python tools/replay.py data/captures --feeding  # the "still at bowl / bowl empty" question
```

Reference photos are sent as a cached prompt prefix (1 h TTL) so they are paid for roughly
once per hour, not per call. The model is `LLM_MODEL` in `config.toml` (default
`claude-opus-5`). Note the cache minimum on `claude-haiku-4-5` is 4096 tokens, so with few
reference photos the prefix silently won't cache on that model.

## Dashboard from your phone

The app runs on the same box as robot-computer, whose dashboard owns 8080, so this one
listens on `127.0.0.1:8081` only. Nothing reaches it except through the host:

- **Cloudflare Tunnel**: `tools/expose.sh <hostname>` adds the hostname as a second
  ingress rule on robot-computer's existing tunnel (backs up and validates the config
  first), so `https://<hostname>/` works from anywhere with no router port open.
- **Tailscale Serve**: the same script runs `tailscale serve --https=8443`, so
  `https://robot-computer.<tailnet>.ts.net:8443/` works from the tailnet.

Every route except `/healthz` asks for HTTP basic auth (`DASH_USER` / `DASH_PASSWORD`);
the phone's browser remembers it. Put Cloudflare Access in front of the public hostname
too, the same as robot-computer, since basic auth has no rate limiting.

Set `DASH_PUBLIC_URL` in `.env` to the public URL so Slack messages link straight to the
event page; the links carry no credentials. The page shows state, plates, feeds this
window, next allowed feed, heartbeat and camera age, LLM calls and estimated cost today,
the **live camera** (MJPEG at the ingest rate, ~2 fps, with the last verdict overlaid and a
pause button for cellular), all controls, a confirm-gated **Feed now**, a confirm-gated
**Reset cycle**, and the event log with frames and LLM JSON. **Reset cycle** is for the night
the window opened before you were in bed and Grrr got fed early: it closes the lid if open,
puts the bowls served tonight back to loaded, and clears the feed count and the minimum-gap
timer so the night starts over. Untick a bowl afterwards if she really did finish it. The live view reuses the frames the app already decodes, so
watching it opens no second connection to the camera and there's no need for the Kasa app.

**Bedtime mode** (`FEED_START = "bedtime"`) is for nights when the first feed should help Grrr
back to sleep, not start while she's still wound up. Feeding waits for a confirm-gated
**We're in bed**, which also does a Reset cycle. After the press the first feed is held until
she has gone `SETTLE_MIN` (60) minutes without a confident Grrr sighting and then comes back;
every confident sighting before that restarts the hour, so circling the feeder after bed
never opens it. Bowie, the cat and unsure frames don't restart it. Later feeds that night
follow the usual rules. With no press there is no feeding, and Slack gets one reminder
`BEDTIME_REMIND_MIN` into the window. The Feeder card shows *waiting for bed*,
*settling 23/60 min*, or *settled*. `FEED_START = "schedule"` keeps the old behaviour.

## Ops

- Slack gets every open/close/veto/feed_failed/none-left/startup/camera event and a quiet
  heartbeat every `HEARTBEAT_MIN`.
- `tools/watchdog.sh` runs from another machine via cron and alerts if the heartbeat is
  older than 10 minutes or the dashboard is unreachable.
- Everything a restart must not forget (state, open timestamps, plates, counters, events)
  is in `data/db/animaleyes.sqlite`. A restart while the lid is open resumes `FEEDING`; it
  never sends `feed` again.
- CI runs ruff + pytest through `patrickisgreat/actions-toolkit` `python-ci`.

## What is POC, and what changes later

POC shortcuts (search the code for `POC`):

- Identification is a cloud LLM call on every motion burst. Fine for one dog and one
  feeder; costs real money and needs the internet.
- The live view is ~2 fps, the rate motion detection is tuned for.
- The camera is found by MAC assuming a /24 home network.
- No chirp/buzzer on veto: no hardware for it.
- "Bowl eaten" is a single LLM judgement on the last frames before closing.
- The dashboard is one inline HTML page behind basic auth; no users or sessions.

**v1**: local model (small classifier fine-tuned on this camera's frames) on a Raspberry
Pi, PIR sensor to gate the camera instead of frame differencing, a buzzer for vetoes, real
sessions on the dashboard.

**v2**: RFID chip on Grrr's collar as the primary identity, vision as the veto/second
factor.
