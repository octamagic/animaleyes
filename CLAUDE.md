# CLAUDE.md — animaleyes

## What is this project?

animaleyes watches a TP-Link Kasa camera mounted over a PetLibro Polar (PLAF109) wet-food
feeder and opens the feeder only for one dog. A state machine turns motion, LLM
identification verdicts, and feeder state into open/close decisions, records everything in
SQLite, notifies Slack, and serves a phone-sized dashboard. Today it is a POC; it is the
foundation for the real system (local model, Pi, PIR, buzzer, RFID).

## ⚠️ Product Invariants — DO NOT VIOLATE

These are physical-world constraints. A wrong open feeds the wrong animal or wastes a
night's food; a double open empties the feeder. Treat them as locked.

1. **Only Grrr is fed.** Grrr is a tiny, old, black schnoodle. Bowie (medium, blonde, lanky)
   and the cat are never fed. Any identification change must keep "unsure" as the safe,
   preferred answer for dark, blurry, or partial frames. Never lower `GRRR_MIN_CONF` or
   `CONFIRMATIONS_REQUIRED` in code to make a test or a night "work"; those are dashboard
   knobs for the human.

2. **Only the state machine touches the feeder.** `animaleyes/machine.py` is the single
   caller of `open_now`, `rotate`, and `close`. The dashboard, tools, and notifiers set
   flags or read state; they never call `petlibro-cli`. `Feed now` on the dashboard is a
   request the machine consumes on its next tick.

3. **A restart can never double-feed.** State, `opened_at`, `last_open_at`, plates, and
   counters are persisted in SQLite *before* the side effect that depends on them. A
   process that dies in `OPENING` or `FEEDING` resumes as `FEEDING` with the lid assumed
   open. `MIN_GAP_MIN` is enforced from the persisted `last_open_at`, not memory.

4. **`DRY_RUN` defaults to true** and must stay the default in `config.toml`, `Settings`,
   and `.env.example`. In dry run the real feeder object is never called.

5. **Every transition is logged with a reason**, and every open/close/veto/failure event
   stores its frames and the LLM JSON. If you add a state or a path to the feeder, it
   writes an event.

6. **`petlibro-cli/` is vendored verbatim** from `~/code/petlibro-cli`. Do not rewrite it
   here; change it there and copy it over.

If you find yourself wanting to relax one of these to fix a problem, stop and pull a
different lever (prompt, thresholds on the dashboard, reference photos) or escalate.

## Tech Stack

- **Language**: Python 3.12, type-hinted, dataclasses and Protocols over frameworks
- **Vision**: Claude via the `anthropic` SDK, prompt caching for the reference photos,
  JSON-schema constrained output
- **Video**: `ffmpeg` subprocess emitting JPEGs; Pillow for motion differencing
- **Web**: FastAPI + uvicorn, one inline HTML page, no build step
- **Storage**: SQLite (stdlib `sqlite3`), `config.toml` via `tomllib`/`tomli-w`
- **Feeder**: `petlibro-cli` (vendored) as a subprocess
- **Runtime**: Docker Compose (host network) on the robot-computer box, dashboard on
  127.0.0.1:8081 behind basic auth, reached via Cloudflare Tunnel or Tailscale Serve
- **Testing**: pytest with fakes for clock, LLM, feeder, notifier
- **CI**: GitHub Actions via `octamagic/actions-toolkit` `python-ci`

## Common Commands

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" -e ./petlibro-cli
.venv/bin/animaleyes                        # run locally (needs .env, ffmpeg)
.venv/bin/pytest                            # unit tests, < 1 s
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python tools/capture.py --once    # prove the camera
.venv/bin/python tools/replay.py data/captures   # prove identification
docker compose up -d --build                # on the box
tools/watchdog.sh                           # from another machine
tools/expose.sh <hostname>                  # once: Cloudflare hostname + Tailscale Serve
```

## Project Structure

```
animaleyes/
├── main.py         # wiring + loop thread + uvicorn; the only place real objects are built
├── machine.py      # THE STATE MACHINE. Only caller of the feeder.
├── config.py       # Settings (config.toml, re-read every tick) and Secrets (.env)
├── store.py        # SQLite: kv, plates, events, llm_calls
├── camera.py       # ffmpeg subprocess -> FrameBuffer; RTSP or legacy Kasa HTTPS
├── motion.py       # frame differencing on a 64x36 greyscale thumbnail
├── vision.py       # Claude identification; reference photos as cached prefix
├── feeder.py       # petlibro-cli wrapper (PetlibroCli) and DryRunFeeder
├── notify.py       # Slack webhook
└── dashboard.py    # FastAPI routes + the inline page
tools/              # capture.py, replay.py, watchdog.sh, expose.sh
tests/              # conftest.py has the fakes and the Harness
petlibro-cli/       # vendored, do not edit here
data/reference/     # {grrr,bowie,cat}/ photos, committed (small set)
data/{captures,frames,db}/   # runtime output, gitignored
config.toml         # runtime knobs, all editable from the dashboard
```

## Conventions

- Inject time (`clock`), the LLM, the feeder, and the notifier. Nothing in `machine.py`
  imports `datetime.now`, `subprocess`, or `anthropic`.
- Persist before you act. Write the kv/plate change, then call the outside world.
- Log events, not prints. `store.add_event(now, kind, reason, data, frames)`.
- Config keys are CAPS in `config.toml` and on `Settings`; secrets are only in `.env`.
- Mark deliberate shortcuts with a `POC` comment so they can be found later.
- Match the surrounding style: small functions, explicit names, no speculative
  abstractions.

## Testing

No PR merges without tests for the behavior it changes. The state machine is tested only
through `tests/conftest.py`'s `Harness` with fakes; add scenarios there, never call the
network. Every scenario named in the product spec has a test in `tests/test_machine.py`;
keep them passing and add one when a rule changes. Dashboard routes are tested with
FastAPI's `TestClient`.

## Git Workflow

- Work from a branch (`feat/...`, `fix/...`, `chore/...`); never commit to `main`.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`, `docs:`).
- Open PRs with `gh pr create`; the user reviews and merges. Do not merge autonomously.
- **Never add `Co-Authored-By` or "Generated with Claude Code" to commits or PRs.**
- Toolkit changes go to the toolkit repo on a branch with a conventional-commit PR title,
  then are consumed here by tag. `actions-toolkit` is pinned to `@main` until its first
  tagged release exists; switch to `@v1` when it does.
